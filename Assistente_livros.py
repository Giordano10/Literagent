# app.py

import os
import streamlit as st
import fitz  # PyMuPDF
import io
import json
import re
from datetime import datetime
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- GOOGLE DRIVE IMPORTS ---
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- CONSTANTES ---
FAISS_INDEX_PATH = "faiss_index"
FAISS_MANIFEST_FILE = "faiss_manifest.json"
CREDENTIALS_FILE = "credentials.json"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1wYDn0Bvscp8zMmIJwf3q-uPT4VowDCU8?usp=drive_link"

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Assistente de Livros", page_icon="📚", layout="wide")

# --- GOOGLE DRIVE & LÓGICA DE DADOS ---

def authenticate_gdrive():
    """Autentica na API do Google Drive usando a conta de serviço."""
    if not os.path.exists(CREDENTIALS_FILE):
        st.error(f"Arquivo de credenciais '{CREDENTIALS_FILE}' não encontrado.")
        return None
    try:
        creds = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        st.error(f"Falha na autenticação com o Google Drive: {e}")
        return None

def get_folder_id_from_url(url):
    """Extrai o ID da pasta de uma URL do Google Drive."""
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

def list_gdrive_files(service, folder_id):
    """Lista os arquivos PDF em uma pasta do Google Drive."""
    try:
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        results = service.files().list(q=query, pageSize=100, fields="nextPageToken, files(id, name, modifiedTime)").execute()
        return {item['id']: {'name': item['name'], 'modified_time': item['modifiedTime']} for item in results.get('files', [])}
    except HttpError as e:
        st.error(f"Erro ao listar arquivos do Google Drive: {e}")
        return {}

def download_gdrive_files_as_streams(service, file_ids):
    """Baixa arquivos do Drive e retorna como streams de bytes."""
    pdf_streams = []
    for file_id in file_ids:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        pdf_streams.append(fh)
    return pdf_streams

def load_manifest():
    """Carrega o manifesto de arquivos processados."""
    if os.path.exists(FAISS_MANIFEST_FILE):
        with open(FAISS_MANIFEST_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_manifest(data):
    """Salva o manifesto de arquivos processados."""
    with open(FAISS_MANIFEST_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- FUNÇÕES CORE (Adaptadas) ---

def get_pdf_text(pdf_docs_streams):
    """Extrai texto de uma lista de streams de PDF."""
    text = ""
    for pdf_stream in pdf_docs_streams:
        with fitz.open(stream=pdf_stream.read(), filetype="pdf") as doc:
            text += "".join(page.get_text() for page in doc)
    return text

def get_text_chunks(text):
    """Divide o texto em chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def get_conversational_rag_chain(vector_store, api_key):
    """Cria a cadeia de conversação RAG."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.3)
    retriever = vector_store.as_retriever()
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Dado um histórico de chat e a última pergunta do usuário, que pode fazer referência ao contexto no histórico, formule uma pergunta independente que possa ser entendida sem o histórico. NÃO responda à pergunta, apenas a reformule se necessário, caso contrário, retorne-a como está."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente especialista... {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: StreamlitChatMessageHistory(key="chat_history"),
        input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer",
    )

# --- INTERFACE DO STREAMLIT ---

if "conversation" not in st.session_state:
    st.session_state.conversation = None

st.header("Assistente de Livros 📚")
st.write("Converse com seus documentos do Google Drive.")

# Carrega a chave de API do ambiente
api_key = os.getenv("GOOGLE_API_KEY")

with st.sidebar:
    st.subheader("Configuração")
    if api_key:
        st.success("API Key do Google carregada do .env!")
    else:
        st.error("API Key do Google não encontrada.")
        st.error("Crie um arquivo .env na raiz do projeto e adicione a linha: GOOGLE_API_KEY='SUA_CHAVE'")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    
    if os.path.exists(FAISS_INDEX_PATH) and st.session_state.conversation is None:
        with st.spinner("Carregando base de conhecimento local..."):
            try:
                vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
                st.session_state.conversation = get_conversational_rag_chain(vector_store, api_key)
                st.sidebar.success("Base de conhecimento local carregada!")
            except Exception as e:
                st.sidebar.error(f"Erro ao carregar base local: {e}")

    st.subheader("Sincronizar com Google Drive")
    st.info("Pasta do Drive definida no código.")

    if st.button("Sincronizar"):
        folder_id = get_folder_id_from_url(GDRIVE_FOLDER_URL)
        if not folder_id:
            st.error("URL da pasta inválida no código.")
        else:
            with st.spinner("Sincronizando com Google Drive..."):
                gdrive_service = authenticate_gdrive()
                if gdrive_service:
                    st.write("1/5 - Listando arquivos...")
                    drive_files = list_gdrive_files(gdrive_service, folder_id)
                    st.write("2/5 - Verificando arquivos...")
                    manifest = load_manifest()
                    files_to_process = {fid: finfo for fid, finfo in drive_files.items() if fid not in manifest or datetime.fromisoformat(finfo['modified_time'][:-1]) > datetime.fromisoformat(manifest[fid]['modified_time'][:-1])}
                    
                    if not files_to_process:
                        st.success("Base de conhecimento já está atualizada!")
                    else:
                        st.write(f"3/5 - Baixando {len(files_to_process)} arquivo(s)...")
                        new_pdf_streams = download_gdrive_files_as_streams(gdrive_service, files_to_process.keys())
                        st.write("4/5 - Processando texto...")
                        new_raw_text = get_pdf_text(new_pdf_streams)
                        new_text_chunks = get_text_chunks(new_raw_text)
                        st.write("5/5 - Atualizando a base...")
                        if os.path.exists(FAISS_INDEX_PATH):
                            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
                            vector_store.add_texts(new_text_chunks)
                        else:
                            vector_store = FAISS.from_texts(new_text_chunks, embedding=embeddings)
                        
                        vector_store.save_local(FAISS_INDEX_PATH)
                        save_manifest(drive_files)
                        st.session_state.conversation = get_conversational_rag_chain(vector_store, api_key)
                        st.success("Base de conhecimento atualizada!")
                        st.rerun()

# --- ÁREA DE CHAT ---
st.subheader("Chat")

if not api_key:
    st.warning("Configure sua API Key no arquivo .env")
elif st.session_state.conversation is None:
    st.info("Sincronize com o Google Drive para começar.")

for msg in st.session_state.get("chat_history", []):
    with st.chat_message(msg.type):
        st.markdown(msg.content)

if user_question := st.chat_input("Qual a sua pergunta?"):
    if st.session_state.conversation:
        with st.chat_message("user"): st.markdown(user_question)
        config = {"configurable": {"session_id": "streamlit_user"}}
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = st.session_state.conversation.invoke({"input": user_question}, config)
                st.markdown(response["answer"])
    else:
        st.warning("Por favor, sincronize com o Google Drive.")
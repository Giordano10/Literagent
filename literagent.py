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

# Carrega as variáveis de ambiente do arquivo .env (para desenvolvimento local)
load_dotenv()

# --- CONSTANTES ---
FAISS_INDEX_PATH = "faiss_index"
FAISS_MANIFEST_FILE = "faiss_manifest.json"
CREDENTIALS_FILE = "credentials.json"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1wYDn0Bvscp8zMmIJwf3q-uPT4VowDCU8?usp=drive_link"

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="LiterAgent", page_icon="📚", layout="wide")

# --- GERENCIAMENTO DE SECRETS ---
def manage_secrets():
    """Gerencia chaves para ambientes locais e de nuvem (Streamlit Cloud)."""
    # Prioriza os segredos do Streamlit Cloud se disponível
    if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        # Se estiver na nuvem, cria o credentials.json a partir dos segredos
        if "gcp_service_account" in st.secrets and not os.path.exists(CREDENTIALS_FILE):
            with open(CREDENTIALS_FILE, "w") as f:
                # Convertendo o objeto de secrets para um dicionário antes de salvar
                credentials_dict = dict(st.secrets["gcp_service_account"])
                json.dump(credentials_dict, f)
        return api_key
    else:
        # Fallback para o arquivo .env (ambiente local)
        return os.getenv("GOOGLE_API_KEY")

# --- LÓGICA DE DADOS ---

def authenticate_gdrive():
    """Autentica na API do Google Drive."""
    if not os.path.exists(CREDENTIALS_FILE):
        st.error(f"Arquivo '{CREDENTIALS_FILE}' não encontrado.")
        return None
    try:
        creds = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        st.error(f"Falha na autenticação com o Google Drive: {e}")
        return None

def get_folder_id_from_url(url):
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

def list_gdrive_files(service, folder_id):
    try:
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        results = service.files().list(q=query, pageSize=100, fields="nextPageToken, files(id, name, modifiedTime)").execute()
        return {item['id']: {'name': item['name'], 'modified_time': item['modifiedTime']} for item in results.get('files', [])}
    except HttpError as e:
        st.error(f"Erro ao listar arquivos do Drive: {e}")
        return {}

def download_gdrive_files_as_streams(service, file_ids):
    streams = []
    for file_id in file_ids:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        streams.append(fh)
    return streams

def load_manifest():
    return json.load(open(FAISS_MANIFEST_FILE, 'r')) if os.path.exists(FAISS_MANIFEST_FILE) else {}

def save_manifest(data):
    with open(FAISS_MANIFEST_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- FUNÇÕES CORE ---

def get_pdf_text(pdf_docs_streams):
    text = ""
    for pdf_stream in pdf_docs_streams:
        with fitz.open(stream=pdf_stream.read(), filetype="pdf") as doc:
            text += "".join(page.get_text() for page in doc)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def get_conversational_rag_chain(vector_store, api_key, temperature):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=temperature)
    retriever = vector_store.as_retriever()
    contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", "... reformule a pergunta ..."), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages([("system", "... Use o contexto para responder.\nContexto: {context}"), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return RunnableWithMessageHistory(rag_chain, lambda s_id: StreamlitChatMessageHistory(key="chat_history"), input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer")

# --- INTERFACE DO STREAMLIT ---

st.header("LiterAgent 📚")
st.write("Converse com seus documentos do Google Drive.")

if "conversation" not in st.session_state:
    st.session_state.conversation = None

api_key = manage_secrets()

with st.sidebar:
    st.subheader("Configuração")
    if api_key:
        st.success("API Key do Google carregada!")
    else:
        st.error("API Key do Google não configurada.")
        st.info("Configure-a no seu arquivo .env ou nos Secrets do Streamlit Cloud.")
        st.stop()

    st.subheader("Ajuste de Criatividade")
    model_temperature = st.slider("Temperatura", 0.0, 1.0, 0.3, 0.05, help="Baixo: factual. Alto: criativo.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            st.session_state.conversation = get_conversational_rag_chain(vector_store, api_key, model_temperature)
        except Exception as e:
            st.error(f"Erro ao carregar base local: {e}")

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
                    drive_files = list_gdrive_files(gdrive_service, folder_id)
                    manifest = load_manifest()
                    files_to_process = {fid: finfo for fid, finfo in drive_files.items() if fid not in manifest or datetime.fromisoformat(finfo['modified_time'][:-1]) > datetime.fromisoformat(manifest[fid]['modified_time'][:-1])}
                    
                    if not files_to_process:
                        st.success("Base de conhecimento já está atualizada!")
                    else:
                        st.write(f"Baixando {len(files_to_process)} novo(s) arquivo(s)...")
                        new_pdf_streams = download_gdrive_files_as_streams(gdrive_service, files_to_process.keys())
                        new_raw_text = get_pdf_text(new_pdf_streams)
                        new_text_chunks = get_text_chunks(new_raw_text)
                        
                        st.write("Atualizando a base de conhecimento...")
                        if os.path.exists(FAISS_INDEX_PATH):
                            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
                            vector_store.add_texts(new_text_chunks)
                        else:
                            vector_store = FAISS.from_texts(new_text_chunks, embedding=embeddings)
                        
                        vector_store.save_local(FAISS_INDEX_PATH)
                        save_manifest(drive_files)
                        st.session_state.conversation = get_conversational_rag_chain(vector_store, api_key, model_temperature)
                        st.success("Base de conhecimento atualizada!")
                        st.rerun()

# --- ÁREA DE CHAT ---
st.subheader("Chat")

if st.session_state.conversation is None:
    st.info("Sincronize com o Google Drive para carregar a base de conhecimento.")

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
        st.warning("A base de conhecimento não está carregada. Sincronize com o Drive.")
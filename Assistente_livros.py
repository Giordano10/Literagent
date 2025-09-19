# app.py

import os
import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Assistente de Livros",
    page_icon="📚",
    layout="wide"
)

# --- FUNÇÕES CORE ---

def get_pdf_text(pdf_docs):
    """Extrai o texto de uma lista de documentos PDF."""
    text = ""
    for pdf in pdf_docs:
        with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
            text += "".join(page.get_text() for page in doc)
    return text

def get_text_chunks(text):
    """Divide o texto em chunks menores."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    """Cria e retorna um vector store a partir dos chunks de texto."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Erro ao criar o vector store: {e}")
        st.error("Verifique sua chave de API ou a conexão com a internet.")
        return None

def get_conversational_rag_chain(vector_store, api_key):
    """Cria a cadeia de conversação RAG com gerenciamento de histórico."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.3)
    retriever = vector_store.as_retriever()

    # 1. Prompt para reformular a pergunta com base no histórico
    contextualize_q_system_prompt = (
        "Dado um histórico de chat e a última pergunta do usuário, "
        "que pode fazer referência ao contexto no histórico, "
        "formule uma pergunta independente que possa ser entendida sem o histórico. "
        "NÃO responda à pergunta, apenas a reformule se necessário, caso contrário, retorne-a como está."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Prompt para responder a pergunta com base no contexto recuperado
    qa_system_prompt = """Você é um assistente especialista e deve responder perguntas com base nos documentos PDF fornecidos.
    Use os trechos de contexto a seguir para responder à pergunta.
    Se você não sabe a resposta ou a informação não está no contexto, diga "Com base nos meus documentos, não encontrei uma resposta para isso.", não tente inventar uma resposta.
    Responda de forma concisa e direta.

    Contexto:
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. Combina o recuperador e a cadeia de resposta
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 4. Adiciona o gerenciamento de histórico
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: StreamlitChatMessageHistory(key="chat_history"),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

# --- INTERFACE DO STREAMLIT ---

# Inicializa o estado da sessão
if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Cabeçalho
st.header("Assistente de Livros 📚")
st.write("Faça perguntas sobre o conteúdo dos seus livros e documentos.")

# Sidebar para upload e configuração
with st.sidebar:
    st.subheader("Seus Documentos")
    
    api_key = st.text_input("Google API Key", type="password", help="Obtenha sua chave no Google AI Studio.")
    
    pdf_docs = st.file_uploader(
        "Carregue seus PDFs aqui e clique em 'Processar'", 
        accept_multiple_files=True,
        type="pdf"
    )

    if st.button("Processar"):
        if not api_key:
            st.warning("Por favor, insira sua Google API Key.")
        elif not pdf_docs:
            st.warning("Por favor, carregue pelo menos um arquivo PDF.")
        else:
            with st.spinner("Processando seus documentos... Isso pode levar um momento."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks, api_key)
                
                if vector_store:
                    st.session_state.conversation = get_conversational_rag_chain(vector_store, api_key)
                    st.success("Documentos processados! Pronto para conversar.")

# Área principal do Chat
st.subheader("Chat")

# Exibe o histórico de mensagens
for msg in st.session_state.get("chat_history", []):
    with st.chat_message(msg.type):
        st.markdown(msg.content)

# Input do usuáriod
if user_question := st.chat_input("Qual a sua pergunta sobre os documentos?"):
    if st.session_state.conversation:
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Define um ID de sessão para o histórico (pode ser qualquer string fixa para um app de usuário único)
        config = {"configurable": {"session_id": "streamlit_user"}}
        
        # Invoca a cadeia e exibe a resposta
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = st.session_state.conversation.invoke({"input": user_question}, config)
                st.markdown(response["answer"])
    else:
        st.warning("Por favor, processe seus PDFs primeiro.")

# Assistente de Livros 📚

Este projeto é um assistente de chat que utiliza a tecnologia RAG (Retrieval-Augmented Generation) para responder a perguntas sobre documentos PDF que você fornece. A interface é construída com Streamlit, e o backend utiliza os modelos de linguagem da Google e a biblioteca LangChain.

## Funcionalidades

- **Upload de Múltiplos PDFs:** Carregue um ou mais documentos PDF para análise.
- **Chat Interativo:** Converse com um assistente de IA que responde com base no conteúdo dos seus documentos.
- **Histórico de Conversa:** O assistente mantém o contexto da conversa para respostas mais precisas.
- **Modelo de Linguagem Google Gemini:** Utiliza o `gemini-2.5-pro` para geração de respostas e `models/text-embedding-004` para a criação de embeddings.

## Dependências

O projeto utiliza as seguintes bibliotecas Python:

```
streamlit
google-generativeai
langchain
langchain-google-genai
langchain-community
pymupdf
chromadb
faiss-cpu
protobuf
```

## Instalação

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd <NOME_DO_DIRETORIO>
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # No Windows, use: .venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure sua chave de API do Google:**
    O assistente requer uma chave de API do Google para funcionar. Você pode obtê-la no [Google AI Studio](https://aistudio.google.com/app/apikey).

    A aplicação irá pedir a chave na interface, ou você pode configurar uma variável de ambiente:
    ```bash
    export GOOGLE_API_KEY="SUA_CHAVE_API"
    ```

## Como Utilizar

### 1. Executando o Assistente de Chat

Após a instalação, inicie a aplicação Streamlit com o seguinte comando:

```bash
streamlit run Assistente_livros.py
```

A interface web será aberta no seu navegador. Siga os passos na barra lateral:
1.  Insira sua Google API Key.
2.  Faça o upload dos seus arquivos PDF.
3.  Clique no botão "Processar".
4.  Após o processamento, você pode começar a fazer perguntas no chat.

### 2. Scripts Auxiliares

O projeto inclui alguns scripts úteis:

-   **`listar_modelos.py`**: Lista todos os modelos da API do Google Generative AI disponíveis para a sua chave.
    ```bash
    python listar_modelos.py
    ```

-   **`verificar_ambiente.py`**: Verifica se a biblioteca `PyMuPDF` (fitz) está corretamente instalada no seu ambiente.
    ```bash
    python verificar_ambiente.py
    ```

### 3. Executando os Testes

O projeto contém testes unitários para as funções principais. Para executá-los:

```bash
python test_assistente_livros.py
```

## Estrutura do Projeto

-   `Assistente_livros.py`: O arquivo principal da aplicação Streamlit.
-   `requirements.txt`: Lista de dependências do projeto.
-   `test_assistente_livros.py`: Testes unitários para o assistente.
-   `listar_modelos.py`: Script para listar os modelos de IA disponíveis.
-   `verificar_ambiente.py`: Script para verificar a instalação do PyMuPDF.
-   `meus_livros/`: Diretório sugerido para armazenar seus PDFs.

# Assistente de Livros 📚

Este projeto é um assistente de chat que utiliza a tecnologia RAG (Retrieval-Augmented Generation) para responder a perguntas sobre documentos PDF armazenados em uma pasta específica do Google Drive.

A interface é construída com Streamlit e o backend utiliza os modelos de linguagem da Google (Gemini) e a biblioteca LangChain para orquestrar o fluxo de dados.

## Funcionalidades Principais

- **Integração com Google Drive:** Conecta-se a uma pasta do Google Drive para usar seus PDFs como base de conhecimento.
- **Sincronização Inteligente:** Processa apenas arquivos novos ou modificados, evitando reprocessamento desnecessário a cada execução.
- **Base de Conhecimento Persistente:** Salva o índice de vetores (FAISS) localmente para um carregamento rápido e eficiente.
- **Chat com Memória:** Mantém o contexto da conversa atual para permitir perguntas de acompanhamento.
- **Segurança:** Mantém as chaves e credenciais fora do controle de versão através do uso de `.gitignore`.

## Arquitetura

O fluxo de dados da aplicação segue os seguintes passos:

1.  **Autenticação:** O Streamlit se conecta à API do Google Drive usando um arquivo de credenciais de conta de serviço (`credentials.json`).
2.  **Sincronização:** Lista os PDFs na pasta do Drive e compara com um manifesto local (`faiss_manifest.json`) para encontrar arquivos novos/modificados.
3.  **Processamento:** Os novos arquivos são baixados, seu texto é extraído (com PyMuPDF) e dividido em `chunks` (com LangChain).
4.  **Embedding e Armazenamento:** Os `chunks` de texto são transformados em vetores (embeddings) pelo modelo `text-embedding-004` da Google e armazenados em um índice FAISS local (`faiss_index`).
5.  **Conversação:** O usuário interage com a aplicação. A pergunta é usada para buscar os `chunks` mais relevantes no índice FAISS, que são então enviados como contexto para o modelo `gemini-2.5-pro` gerar uma resposta.

---

## Configuração Obrigatória

Antes de executar a aplicação, são necessários dois tipos de chaves do Google.

### 1. API Key do Google AI

Esta chave é para usar os modelos de linguagem (Gemini). 

- **Como obter:** Vá ao [Google AI Studio](https://aistudio.google.com/app/apikey) e crie uma nova chave de API.
- **Onde usar:** Você irá inseri-la diretamente na interface do aplicativo.

### 2. Credenciais para o Google Drive

Esta chave permite que a aplicação leia os arquivos da sua pasta no Google Drive de forma segura e autônoma.

**Passo A: Habilitar a API do Google Drive**

1.  Acesse o [Console do Google Cloud](https://console.cloud.google.com/apis/library/drive.googleapis.com).
2.  Selecione um projeto (ou crie um novo).
3.  Verifique se a **"Google Drive API"** está ativada. Se não estiver, clique em **"Ativar"**.

**Passo B: Criar uma Conta de Serviço e a Chave JSON**

1.  No menu do Console, vá para **"APIs e Serviços" > "Credenciais"**.
2.  Clique em **"+ CRIAR CREDENCIAIS"** e selecione **"Conta de serviço"**.
3.  Dê um nome para a conta (ex: `assistente-livros-agent`) e clique em **"CRIAR E CONTINUAR"**.
4.  Pode pular a etapa de "função" clicando em **"CONTINUAR"** e depois em **"CONCLUÍDO"**.
5.  Na lista de credenciais, encontre a conta que você criou e clique nela.
6.  Vá para a aba **"CHAVES"**, clique em **"ADICIONAR CHAVE" > "Criar nova chave"**.
7.  Selecione **JSON** e clique em **"CRIAR"**. Um arquivo JSON será baixado.

**Passo C: Posicionar a Chave e Compartilhar a Pasta**

1.  Renomeie o arquivo JSON baixado para `credentials.json`.
2.  Mova este arquivo para a raiz do projeto (a mesma pasta onde está o `Assistente_livros.py`).
3.  **Passo Crucial:** Abra o `credentials.json` em um editor de texto e copie o email que está no campo `"client_email"`.
4.  Vá até a sua pasta no Google Drive, clique em **"Compartilhar"** e cole este email, garantindo que ele tenha, no mínimo, permissão de **"Leitor"**.

---

## Instalação e Uso

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd <NOME_DO_DIRETORIO>
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv .venv
    # No Windows:
    .venv\Scripts\activate
    # No Linux/macOS:
    # source .venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a aplicação:**
    ```bash
    streamlit run Assistente_livros.py
    ```

5.  **Na interface do aplicativo:**
    -   A aplicação irá carregar a API Key automaticamente do seu arquivo `.env`.
    -   Clique no botão **"Sincronizar"**.
    -   Aguarde o processamento e comece a conversar!

## Estrutura do Projeto

-   `Assistente_livros.py`: O arquivo principal da aplicação Streamlit.
-   `requirements.txt`: Lista de dependências do projeto.
-   `.gitignore`: Arquivo para ignorar arquivos sensíveis e desnecessários.
-   `credentials.json`: (Ignorado pelo Git) Chave de acesso para a API do Google Drive.
-   `faiss_index/`: (Ignorado pelo Git) Pasta onde o índice de vetores é salvo.
-   `faiss_manifest.json`: (Ignorado pelo Git) Registro dos arquivos já processados.
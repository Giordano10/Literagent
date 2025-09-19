# LiterAgent 📚

Este projeto é um assistente de chat que utiliza a tecnologia RAG (Retrieval-Augmented Generation) para responder a perguntas sobre documentos PDF armazenados em uma pasta específica do Google Drive.

A interface é construída com Streamlit e o backend utiliza os modelos de linguagem da Google (Gemini) e a biblioteca LangChain para orquestrar o fluxo de dados.

## Funcionalidades Principais

- **Integração com Google Drive:** Conecta-se a uma pasta do Google Drive para usar seus PDFs como base de conhecimento.
- **Sincronização Inteligente:** Processa apenas arquivos novos ou modificados, evitando reprocessamento desnecessário a cada execução.
- **Base de Conhecimento Persistente:** Salva o índice de vetores (FAISS) localmente para um carregamento rápido e eficiente.
- **Chat com Memória:** Mantém o contexto da conversa atual para permitir perguntas de acompanhamento.
- **Ajuste de Criatividade:** Controle a "temperatura" do modelo com um slider para obter respostas mais factuais ou mais criativas.
- **Segurança:** Mantém as chaves e credenciais fora do controle de versão através do uso de `.env` e `.gitignore`.
- **Docker-ready:** Inclui um `Dockerfile` para fácil portabilidade e implantação.

## Arquitetura

O fluxo de dados da aplicação segue os seguintes passos:

1.  **Autenticação:** O Streamlit se conecta à API do Google Drive usando um arquivo de credenciais de conta de serviço (`credentials.json`).
2.  **Sincronização:** Lista os PDFs na pasta do Drive e compara com um manifesto local (`faiss_manifest.json`) para encontrar arquivos novos/modificados.
3.  **Processamento:** Os novos arquivos são baixados, seu texto é extraído (com PyMuPDF) e dividido em `chunks` (com LangChain).
4.  **Embedding e Armazenamento:** Os `chunks` de texto são transformados em vetores (embeddings) pelo modelo `text-embedding-004` da Google e armazenados em um índice FAISS local (`faiss_index`).
5.  **Conversação:** O usuário interage com a aplicação. A pergunta é usada para buscar os `chunks` mais relevantes no índice FAISS, que são então enviados como contexto para o modelo `gemini-1.5-pro` gerar uma resposta.

---

## Configuração Obrigatória

Antes de executar a aplicação, são necessários dois tipos de chaves do Google.

### 1. API Key do Google AI (via .env)

Esta chave é para usar os modelos de linguagem (Gemini) e é carregada de forma segura através de um arquivo de ambiente.

- **Como obter:** Vá ao [Google AI Studio](https://aistudio.google.com/app/apikey) e crie uma nova chave de API.
- **Onde usar:**
    1.  Crie um arquivo chamado `.env` na raiz do projeto.
    2.  Dentro dele, adicione a seguinte linha, substituindo `SUA_CHAVE_AQUI` pela sua chave real:
        ```
        GOOGLE_API_KEY="SUA_CHAVE_AQUI"
        ```
O arquivo `.gitignore` já está configurado para impedir que este arquivo seja enviado ao seu repositório.

### 2. Credenciais para o Google Drive

Esta chave permite que a aplicação leia os arquivos da sua pasta no Google Drive de forma segura e autônoma.

**Passo A: Habilitar a API do Google Drive**

1.  Acesse o [Console do Google Cloud](https://console.cloud.google.com/apis/library/drive.googleapis.com).
2.  Selecione um projeto (ou crie um novo).
3.  Verifique se a **"Google Drive API"** está ativada. Se não estiver, clique em **"Ativar"**.

**Passo B: Criar uma Conta de Serviço e a Chave JSON**

1.  No menu do Console, vá para **"APIs e Serviços" > "Credenciais"**.
2.  Clique em **"+ CRIAR CREDENCIAIS"** e selecione **"Conta de serviço"**.
3.  Dê um nome para a conta (ex: `literagent-service-account`) e clique em **"CRIAR E CONTINUAR"**.
4.  Pode pular a etapa de "função" clicando em **"CONTINUAR"** e depois em **"CONCLUÍDO"**.
5.  Na lista de credenciais, encontre a conta que você criou e clique nela.
6.  Vá para a aba **"CHAVES"**, clique em **"ADICIONAR CHAVE" > "Criar nova chave"**.
7.  Selecione **JSON** e clique em **"CRIAR"**. Um arquivo JSON será baixado.

**Passo C: Posicionar a Chave e Compartilhar a Pasta**

1.  Renomeie o arquivo JSON baixado para `credentials.json`.
2.  Mova este arquivo para a raiz do projeto.
3.  **Passo Crucial:** Abra o `credentials.json` e copie o email do campo `"client_email"`.
4.  Vá até a sua pasta no Google Drive, clique em **"Compartilhar"** e cole este email, garantindo permissão de **"Leitor"**.

---

## Instalação e Uso Local

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
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a aplicação:**
    ```bash
    streamlit run literagent.py
    ```

5.  **Na interface do aplicativo:**
    -   A aplicação irá carregar a API Key automaticamente do seu arquivo `.env`.
    -   Na barra lateral, ajuste a **Temperatura do Modelo** se desejar.
    -   Clique no botão **"Sincronizar"** para carregar/atualizar seus documentos.
    -   Aguarde o processamento e comece a conversar!

---

## Executando com Docker

Depois de seguir a **Configuração Obrigatória** (criar os arquivos `.env` e `credentials.json`), você pode construir e executar a aplicação em um contêiner Docker.

### 1. Construindo a Imagem

Na raiz do projeto, execute o comando a seguir para construir a imagem. A tag `-t literagent` nomeia a imagem para facilitar o uso.

```bash
docker build -t literagent .
```

### 2. Executando o Contêiner

Para executar o contêiner, você precisa passar suas credenciais de forma segura. O comando a seguir faz isso:

-   `-p 8501:8501`: Mapeia a porta do seu computador para a porta do contêiner.
-   `--env-file .env`: Passa todas as variáveis (sua `GOOGLE_API_KEY`) do seu arquivo `.env` para o contêiner.
-   `-v .../credentials.json:/app/credentials.json:ro`: Monta o seu arquivo `credentials.json` local dentro do contêiner em modo somente leitura (`:ro`).

**Comando para Windows (usando PowerShell):**
```powershell
docker run -p 8501:8501 --env-file .env -v ${PWD}/credentials.json:/app/credentials.json:ro literagent
```

**Comando para Linux/macOS:**
```bash
docker run -p 8501:8501 --env-file .env -v "$(pwd)/credentials.json:/app/credentials.json:ro" literagent
```

Após executar o comando, acesse `http://localhost:8501` no seu navegador.

## Estrutura do Projeto

-   `literagent.py`: O arquivo principal da aplicação Streamlit.
-   `Dockerfile`: Receita para construir a imagem Docker da aplicação.
-   `requirements.txt`: Lista de dependências do projeto.
-   `.gitignore`: Arquivo para ignorar arquivos sensíveis na submissão para o Git.
-   `.dockerignore`: Arquivo para ignorar arquivos sensíveis na construção da imagem Docker.
-   `.env`: (Ignorado pelo Git) Arquivo para armazenar a `GOOGLE_API_KEY`.
-   `credentials.json`: (Ignorado pelo Git) Chave de acesso para a API do Google Drive.
-   `faiss_index/`: (Ignorado pelo Git) Pasta onde o índice de vetores é salvo.
-   `faiss_manifest.json`: (Ignorado pelo Git) Registro dos arquivos já processados.
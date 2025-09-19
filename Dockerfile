# Use uma imagem Python oficial como base
FROM python:3.13-slim

# Define variáveis de ambiente
ENV PYTHONUNBUFFERED=1

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia o arquivo de dependências para o diretório de trabalho
COPY requirements.txt .

# Instala as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante dos arquivos da aplicação para o diretório de trabalho
# O .dockerignore garantirá que arquivos desnecessários não sejam copiados
COPY . .

# Expõe a porta que o Streamlit usa por padrão
EXPOSE 8501

# Define o comando para executar a aplicação quando o contêiner iniciar
CMD ["streamlit", "run", "literagent.py"]

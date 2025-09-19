import google.generativeai as genai
import os
from getpass import getpass

# Tenta obter a chave da variável de ambiente, se não, pede ao usuário
try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
except KeyError:
    GOOGLE_API_KEY = getpass('Cole sua Google API Key: ')

genai.configure(api_key=GOOGLE_API_KEY)

print("--- Modelos Disponíveis para sua API Key ---")

# Listas para organizar os modelos por tipo de tarefa
modelos_geracao = []
modelos_embedding = []
outros_modelos = []

for model in genai.list_models():
    # Verifica se o modelo suporta geração de conteúdo (chat/texto)
    if 'generateContent' in model.supported_generation_methods:
        modelos_geracao.append(model.name)
    # Verifica se o modelo suporta criação de embeddings
    elif 'embedContent' in model.supported_generation_methods:
        modelos_embedding.append(model.name)
    else:
        outros_modelos.append(model.name)

# Imprime os resultados de forma organizada
if modelos_geracao:
    print("\n✅ Modelos de Geração de Conteúdo (Chat/Texto):")
    for m in sorted(modelos_geracao):
        print(f"   - {m}")

if modelos_embedding:
    print("\n✨ Modelos de Embedding (para RAG):")
    for m in sorted(modelos_embedding):
        print(f"   - {m}")

if outros_modelos:
    print("\n🔧 Outros Modelos:")
    for m in sorted(outros_modelos):
        print(f"   - {m}")
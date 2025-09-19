# verificar_ambiente.py
import sys
import pkg_resources

print("--- Verificação de Ambiente Python ---")
print(f"Caminho do Executável Python: {sys.executable}")
print("-" * 35)

try:
    dist = pkg_resources.get_distribution('PyMuPDF')
    print(f"PyMuPDF encontrado!")
    print(f"  - Versão: {dist.version}")
    print(f"  - Localização: {dist.location}")
except pkg_resources.DistributionNotFound:
    print("PyMuPDF (fitz) NÃO FOI ENCONTRADO neste ambiente.")

print("-" * 35)
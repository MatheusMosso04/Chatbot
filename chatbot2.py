import warnings
import re
import json

warnings.filterwarnings("ignore")

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


print("🔄  Carregando base de dados de filmes...")

with open("filmes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [Document(page_content=item["content"]) for item in data]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)


print("\n🎥 Bem-vindo ao Chatbot de Recomendações de Filmes!\n")

while True:
    genero_input = input("➡️  Qual gênero de filme você deseja? (Ex.: Ação, Comédia, Drama, etc): ").strip()
    if genero_input:
        break
    print("⚠️ Por favor, informe um gênero válido.\n")

generos_usuario = [g.strip().lower() for g in re.split(r"[,/]| e | com |\band\b|\&", genero_input)]

ano = input("➡️  Existe algum ano de preferência? (Informe o ano ou pressione Enter para ignorar): ").strip()
duracao = input("➡️  Qual duração em minutos? (Informe apenas o número ou pressione Enter para ignorar): ").strip()

consulta = f"Filme com gênero(s) {', '.join(generos_usuario)}"
if ano:
    consulta += f" do ano {ano}"
if duracao:
    consulta += f" com duração aproximadamente {duracao} minutos"

print("\n🔎 Buscando a melhor recomendação de filme para você...\n")
resultados = db.similarity_search(consulta, k=5)


def extrair_info(texto):
    padrao = re.compile(
        r"Nome:\s*(.*?)\n"
        r"\s*Gênero:\s*(.*?)\n"
        r"\s*Ano:\s*(.*?)\n"
        r"\s*Duração:\s*(.*?)\n"
        r"\s*Diretor:\s*(.*?)\n"
        r"\s*Sinopse:\s*(.*?)(?:\n|$)"
    )
    match = padrao.search(texto)
    if match:
        return {
            "nome": match.group(1).strip(),
            "genero": match.group(2).strip(),
            "ano": match.group(3).strip(),
            "duracao": match.group(4).strip(),
            "diretor": match.group(5).strip(),
            "sinopse": match.group(6).strip(),
        }
    return None

filme_encontrado = None
melhor_alternativa = None

for doc in resultados:
    info = extrair_info(doc.page_content)
    if not info:
        continue

    generos_filme = [g.strip().lower() for g in info['genero'].split(",")]
    genero_ok = any(g in generos_filme for g in generos_usuario)
    ano_ok = True if not ano else ano.strip() == info['ano'].strip()

    duracao_ok = True
    if duracao:
        try:
            dur_usuario = int(duracao)
            dur_filme = int(re.findall(r'\d+', info['duracao'])[0])
            duracao_ok = abs(dur_usuario - dur_filme) <= 15
        except:
            duracao_ok = True

    if genero_ok and ano_ok and duracao_ok:
        filme_encontrado = info
        break

    if genero_ok and not melhor_alternativa:
        melhor_alternativa = info

def mostrar_filme(info, aviso=None):
    if aviso:
        print(f"\n {aviso}")
    print("\n🎬 Filme Recomendado:\n")
    print(f"Nome: {info['nome']}")
    print(f"Gênero: {info['genero']}")
    print(f"Ano: {info['ano']}")
    print(f"Duração: {info['duracao']} minutos")
    print(f"Diretor: {info['diretor']}")
    print(f"Sinopse: {info['sinopse']}")

if filme_encontrado:
    mostrar_filme(filme_encontrado)
elif melhor_alternativa:
    mostrar_filme(melhor_alternativa, aviso="Nenhum filme corresponde exatamente a todos os critérios, mas encontramos uma sugestão próxima baseada no que deseja.")
else:
    print("❌  Nenhum filme encontrado que atenda aos critérios informados.")

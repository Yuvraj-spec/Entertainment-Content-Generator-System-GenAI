from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="vector_store/chroma_db",
    embedding_function=embedding_model
)

results = vector_db.similarity_search("airport scene", k=3)

for r in results:
    print("\n--- RESULT ---\n")
    print(r.page_content[:300])
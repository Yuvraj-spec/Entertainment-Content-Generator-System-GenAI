import json
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


DATA_PATH = "data/moviesum_scene_chunks.json"
PERSIST_DIR = "vector_store/chroma_db"

BATCH_SIZE = 500


print("Loading scene chunks...")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    scenes = json.load(f)

print("Total scenes:", len(scenes))


print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)


documents = []

for scene in tqdm(scenes, desc="Preparing documents"):

    doc = Document(
        page_content=scene["text"],
        metadata=scene["metadata"]
    )

    documents.append(doc)


print("Starting batch insertion...")

for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Inserting batches"):

    batch = documents[i:i+BATCH_SIZE]

    vector_db.add_documents(batch)


print("Persisting database...")

vector_db.persist()

print("Vector database created successfully.")
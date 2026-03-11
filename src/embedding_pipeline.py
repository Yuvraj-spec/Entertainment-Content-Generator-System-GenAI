import json
from langchain_huggingface import HuggingFaceEmbeddings

file_path = "data/moviesum_scene_chunks.json"

with open(file_path, "r", encoding="utf-8") as f:
    scenes = json.load(f)

texts = [s["text"] for s in scenes]
metadata = [s["metadata"] for s in scenes]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Total chunks:", len(texts))
print("Embedding model loaded")
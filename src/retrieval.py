import json
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Load scenes
with open("data/moviesum_scene_chunks.json", "r", encoding="utf-8") as f:
    scenes = json.load(f)

texts = [scene["text"] for scene in scenes]

# Tokenize for BM25
tokenized_corpus = [t.split() for t in texts]

bm25 = BM25Okapi(tokenized_corpus)


# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load vector database
vector_db = Chroma(
    persist_directory="vector_store/chroma_db",
    embedding_function=embedding_model
)


def hybrid_search(query, k=3):

    # Vector search
    vector_results = vector_db.similarity_search(query, k=k)

    # BM25 search
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_results = [texts[i] for i in top_bm25_idx]

    # Merge results
    combined = []

    for r in vector_results:
        combined.append(r.page_content)

    for r in bm25_results:
        combined.append(r)

    # Remove duplicates
    unique = list(dict.fromkeys(combined))

    return unique[:k]


if __name__ == "__main__":

    query = "fight scene in a bar"

    results = hybrid_search(query)

    for r in results:
        print("\n--- SCENE ---\n")
        print(r[:400])
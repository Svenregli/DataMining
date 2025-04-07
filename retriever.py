# retriever.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Load model and Chroma collection
embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

client = chromadb.Client(Settings(
    persist_directory="embeddings",
    anonymized_telemetry=False
))
collection = client.get_or_create_collection("papers")

def semantic_search(query: str, k: int = 5):
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    hits = []
    for i in range(len(results["documents"][0])):
        hits.append({
            "title": results["metadatas"][0][i]["title"],
            "url": results["metadatas"][0][i]["url"],
            "independent": results["metadatas"][0][i]["independent"],
            "dependent": results["metadatas"][0][i]["dependent"],
            "abstract": results["documents"][0][i]
        })
    return hits

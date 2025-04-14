import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Load data
df = pd.read_parquet("data/arxiv_raw.parquet")

# Init embedding model
embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Connect to running Chroma server
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection("paper_chunks")

# Chunking logic (simple — one chunk for now)
def chunk_text(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

# Embed and store
for i, row in df.iterrows():
    text_chunks = chunk_text(row["summary"])
    for j, chunk in enumerate(text_chunks):
        embedding = embed_model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{i}_chunk_{j}"],
            metadatas=[{
                "title": row["title"],
                "url": row["pdf_url"],
                "paper_id": str(i),
                "chunk_id": j
            }]
        )

print("✅ Chunks embedded and stored using ChromaDB 1.0+")

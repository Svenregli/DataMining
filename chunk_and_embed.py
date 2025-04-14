import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import datetime 
import numpy as np
# Load data
df = pd.read_parquet("data/arxiv_raw.parquet")

# Init embedding model
embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Connect to running Chroma server
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection("paper_chunks")

# Chunking logic (simple — one chunk for now)
def chunk_text(text, size=1500, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

# Embed and store
for i, row in df.iterrows():
    text_chunks = chunk_text(row["summary"])
    
    # Convert published date to year
    year = pd.to_datetime(row["published"]).year
    raw_authors = row["authors"]
if isinstance(raw_authors, (list, np.ndarray)):
    authors = ", ".join(map(str, raw_authors))
else:
    authors = str(raw_authors)
    for j, chunk in enumerate(text_chunks):
        embedding = embed_model.encode(chunk).tolist()

        metadata = {
            "title": row["title"],
            "url": row["pdf_url"],
            "authors": authors,
            "year": year,
            "category": row["category"],
            "paper_id": str(i),
            "chunk_id": j
        }

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{i}_chunk_{j}"],
            metadatas=[metadata]
        )

print("Chunks embedded and stored using ChromaDB 1.0+")

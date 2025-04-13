# chunk_and_embed.py
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import textwrap

# Config
CHUNK_SIZE = 500  # Approx characters
CHUNK_OVERLAP = 100
MODEL_NAME = "paraphrase-MiniLM-L6-v2"
DB_DIR = "embeddings_chunks"

# Load data
df = pd.read_parquet("data/arxiv_with_variables.parquet")

# Load model
embed_model = SentenceTransformer(MODEL_NAME)

# Set up ChromaDB
chroma_client = chromadb.Client(Settings(
    persist_directory=DB_DIR,
    anonymized_telemetry=False
))
collection = chroma_client.get_or_create_collection(name="paper_chunks")

def chunk_text(text: str, size: int, overlap: int):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

# Add all chunks to DB
for paper_id, row in df.iterrows():
    text = row["summary"]  # Later you can use full text here
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    for i, chunk in enumerate(chunks):
        embedding = embed_model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            metadatas=[{
                "title": row["title"],
                "url": row["pdf_url"],
                "paper_id": str(paper_id),
                "chunk_id": str(i),
                "independent": str(row.get("independent", "")),
                "dependent": str(row.get("dependent", "")),
            }],
            ids=[f"{paper_id}_chunk_{i}"],
            embeddings=[embedding]
        )

print("✅ All chunks embedded and stored in ChromaDB!")

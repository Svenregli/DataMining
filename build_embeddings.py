import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

# Load data
df = pd.read_parquet("data/arxiv_with_variables.parquet")

# Load model
embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Set up ChromaDB
chroma_client = chromadb.Client(Settings(
    persist_directory="embeddings",
    anonymized_telemetry=False
))
collection = chroma_client.get_or_create_collection(name="papers")

# Loop and add to Chroma
for i, row in df.iterrows():
    abstract = row["summary"]
    metadata = {
        "title": row["title"],
        "url": row["pdf_url"],
        "independent": str(row.get("independent", "")),
        "dependent": str(row.get("dependent", ""))
    }
    embedding = embed_model.encode(abstract).tolist()

    collection.add(
        documents=[abstract],
        metadatas=[metadata],
        ids=[f"paper_{i}"],
        embeddings=[embedding]
    )

print("Embeddings generated and saved!")

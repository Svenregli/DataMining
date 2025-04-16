import pandas as pd
import chromadb
import datetime 
import numpy as np
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os

# Load data
df = pd.read_parquet("data/arxiv_raw.parquet")

# Init embedding model
openai_ef = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to running Chroma server
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection("paper_chunks", embedding_function=openai_ef)


# Chunking logic
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
            ids=[f"{i}_chunk_{j}"],
            metadatas=[metadata]
        )

print("Chunks embedded and stored using ChromaDB 1.0+")

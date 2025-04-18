import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import time
import random
import requests
import streamlit as st

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VECTOR_STORE_PATH = "data/vector_store.json"

# Step 1: Embed and store paper chunks
def embed_and_store_chunks(chunks, metadata_list, store_path=VECTOR_STORE_PATH):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    embeddings = [e.embedding for e in response.data]

    records = []
    for chunk, metadata, vector in zip(chunks, metadata_list, embeddings):
        records.append({
            "text": chunk,
            "vector": vector,
            "metadata": metadata
        })

    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"✅ Stored {len(records)} chunks to {store_path}")


# Step 2: Search from stored embeddings
def search_openai_vector_store(query, top_k=5, store_path=VECTOR_STORE_PATH):
    if not os.path.exists(store_path):
        return []

    with open(store_path, "r") as f:
        records = json.load(f)

    query_embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    vectors = np.array([r["vector"] for r in records])
    similarities = cosine_similarity([query_embedding], vectors)[0]

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [(records[i]["text"], records[i]["metadata"]) for i in top_indices]


# Convenience: Replace Chroma-style wrappers
def embed_papers_to_openai(papers, text_splitter):
    chunks = []
    metadatas = []

    for i, paper in enumerate(papers):
        title = paper.get("title") or f"Paper {i}"
        abstract = paper.get("abstract") or ""
        source_url = paper.get("url") or ""

        full_text = f"{title}\n\n{abstract}"
        paper_chunks = text_splitter.split_text(full_text)

        chunks.extend(paper_chunks)
        metadatas.extend([{
            "title": title,
            "source_url": source_url,
            "paper_index": i
        }] * len(paper_chunks))

    embed_and_store_chunks(chunks, metadatas)


def search_chunks(query, k=6):
    return search_openai_vector_store(query=query, top_k=k)




VECTOR_STORE_PATH = "data/vector_store.json"
CACHE_DIR = "data/semantic_scholar_cache"


def load_vector_store_papers(store_path=VECTOR_STORE_PATH):
    if not os.path.exists(store_path):
        return {}

    with open(store_path, "r") as f:
        records = json.load(f)

    unique_papers = {}
    for r in records:
        meta = r.get("metadata", {})
        pid = meta.get("paperId") or meta.get("source_url") or meta.get("title")
        if pid and pid not in unique_papers:
            unique_papers[pid] = meta
    return unique_papers


def enrich_chroma_papers_with_references():
    os.makedirs(CACHE_DIR, exist_ok=True)
    unique_papers = load_vector_store_papers()

    print(f"🔍 Found {len(unique_papers)} unique papers in vector store.")

    base_url = "https://api.semanticscholar.org/graph/v1"
    enriched_count = 0

    for paper_id, meta in unique_papers.items():
        if not paper_id:
            continue

        details_path = os.path.join(CACHE_DIR, f"paper_{paper_id}.json")
        if os.path.exists(details_path):
            continue

        try:
            time.sleep(random.uniform(0.4, 0.7))
            res = requests.get(
                f"{base_url}/paper/{paper_id}",
                params={"fields": "references.title,references.authors,references.year,references.url,references.paperId"}
            )
            res.raise_for_status()
            references = res.json().get("references", [])
            with open(details_path, "w") as f:
                json.dump(references, f, indent=2)
            enriched_count += 1
            print(f"✅ Enriched: {paper_id} with {len(references)} references.")
        except Exception as e:
            print(f"❌ Failed for {paper_id}: {e}")

    return enriched_count


def load_all_cached_papers(folder=CACHE_DIR):
    all_references = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            path = os.path.join(folder, file)
            try:
                with open(path, "r") as f:
                    refs = json.load(f)
                    all_references.append({
                        "paperId": file.replace("paper_", "").replace(".json", ""),
                        "references": refs
                    })
            except Exception as e:
                st.warning(f"Failed to load {file}: {e}")
    return all_references

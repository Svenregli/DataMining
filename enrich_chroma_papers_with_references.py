import os
import json
import requests
import time
import random
import streamlit as st

def load_vector_store_papers(store_path="data/vector_store.json"):
    if not os.path.exists(store_path):
        return []

    with open(store_path, "r") as f:
        records = json.load(f)

    # Deduplicate by paper index or source_url
    unique_papers = {}
    for r in records:
        meta = r.get("metadata", {})
        pid = meta.get("paperId") or meta.get("source_url") or meta.get("title")
        if pid and pid not in unique_papers:
            unique_papers[pid] = meta
    return unique_papers


def enrich_openai_vector_papers_with_references():
    cache_dir = "data/semantic_scholar_cache"
    os.makedirs(cache_dir, exist_ok=True)

    unique_papers = load_vector_store_papers()
    print(f"🔍 Found {len(unique_papers)} unique papers in vector store.")

    base_url = "https://api.semanticscholar.org/graph/v1"

    for paper_id, meta in unique_papers.items():
        if not paper_id:
            continue

        cache_file = os.path.join(cache_dir, f"paper_{paper_id}.json")
        if os.path.exists(cache_file):
            print(f"✅ Already enriched: {paper_id}")
            continue

        try:
            time.sleep(random.uniform(0.4, 0.7))
            res = requests.get(
                f"{base_url}/paper/{paper_id}",
                params={"fields": "references.title,references.authors,references.year,references.url,references.paperId"}
            )
            res.raise_for_status()
            references = res.json().get("references", [])
            with open(cache_file, "w") as f:
                json.dump(references, f, indent=2)
            print(f"✅ Fetched references for {paper_id} ({len(references)} refs)")
        except Exception as e:
            print(f"❌ Failed for {paper_id}: {e}")

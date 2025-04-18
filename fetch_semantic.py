import requests
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import os
import tempfile
import pandas as pd
import requests
import json
import random
import time

# Set up embedding + ChromaDB
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection("semantic_scholar", embedding_function=openai_ef)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 1. Query Semantic Scholar API
import os
import time
import requests
import pandas as pd
import json
import random

def query_semantic_scholar(query: str, limit: int = 10, fetch_references: bool = True):
    base_url = "https://api.semanticscholar.org/graph/v1"
    cache_dir = "data/semantic_scholar_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    last_request_time = [0]

    def make_request_with_backoff(url, params=None, max_retries=5):
        current_time = time.time()
        time_since_last = current_time - last_request_time[0]
        if time_since_last < 1.0:
            time.sleep(1.0 - time_since_last)

        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                last_request_time[0] = time.time()
                return response
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit hit. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    continue
                raise
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
                if attempt < max_retries - 1:
                    continue
                raise
        raise Exception(f"Failed after {max_retries} retries")

    cache_file = os.path.join(cache_dir, f"search_{query.replace(' ', '_')}.json")
    papers = []

    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        try:
            with open(cache_file, 'r') as f:
                papers = json.load(f)
            print(f"Loaded {len(papers)} papers from cache")
        except json.JSONDecodeError:
            print(f"Cache file {cache_file} is corrupted, will fetch fresh data")
            os.remove(cache_file)
            papers = []

    if not papers:
        try:
            search_response = make_request_with_backoff(
                f"{base_url}/paper/search",
                params={
                    "query": query,
                    "limit": limit,
                    "fields": "title,abstract,authors,year,venue,fieldsOfStudy,citationCount,referenceCount,isOpenAccess,openAccessPdf,url,externalIds,paperId"
                }
            )
            papers = search_response.json().get("data", [])
            with open(cache_file, 'w') as f:
                json.dump(papers, f)
            print(f"Fetched and cached {len(papers)} papers")
        except Exception as e:
            print(f"Error in initial search: {e}")
            papers = []

    enriched_results = []
    batch_size = 3

    for i in range(0, len(papers), batch_size):
        batch = papers[i:i+batch_size]

        for paper in batch:
            paper_id = paper.get("paperId")
            references = []

            if fetch_references and paper_id:
                details_cache_file = os.path.join(cache_dir, f"paper_{paper_id}.json")
                references_raw = []

                if os.path.exists(details_cache_file) and os.path.getsize(details_cache_file) > 0:
                    try:
                        with open(details_cache_file, 'r') as f:
                            references_raw = json.load(f)
                        print(f"Loaded references for {paper_id} from cache")
                    except json.JSONDecodeError:
                        print(f"Cache file for {paper_id} is corrupted, will fetch fresh data")
                        os.remove(details_cache_file)
                        references_raw = []

                if not references_raw:
                    try:
                        details_response = make_request_with_backoff(
                            f"{base_url}/paper/{paper_id}",
                            params={
                                "fields": "references.title,references.authors,references.year,references.url"
                            }
                        )
                        references_raw = details_response.json().get("references", [])
                        with open(details_cache_file, 'w') as f:
                            json.dump(references_raw, f)
                    except Exception as e:
                        print(f"Error fetching references for {paper_id}: {e}")
                        references_raw = []

                references = [
                    {
                        "title": ref.get("title"),
                        "year": ref.get("year"),
                        "url": ref.get("url"),
                        "authors": [a.get("name") for a in ref.get("authors", [])] if ref.get("authors") else []
                    }
                    for ref in references_raw
                ]

            else:
                references = []  # ⬅️ Skip references if fetch_references is False

            enriched_results.append({
                "paperId": paper.get("paperId"),
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "url": paper.get("url"),
                "references": references,
                "topics": [],
                "independent_variables": [],
                "dependent_variables": []
            })

        if i + batch_size < len(papers):
            delay = random.uniform(3, 5)
            print(f"Processed {i+batch_size}/{len(papers)} papers. Pausing for {delay:.2f} seconds...")
            time.sleep(delay)

    df = pd.DataFrame(enriched_results)
    os.makedirs("data", exist_ok=True)
    df.to_json("data/semantic_scholar_results.json", orient="records", indent=2)

    return df



# 2. Helper to fetch full PDF text


def fetch_pdf_text(pdf_url):
    response = requests.get(pdf_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    doc = fitz.open(tmp_path)
    return "\n".join([page.get_text() for page in doc])



# 3. Embed papers into ChromaDB
def embed_papers_to_chroma(papers):
    for i, paper in enumerate(papers):
        title = paper.get("title", "No Title")
        abstract = paper.get("abstract", "")
        paper_id = paper.get("paperId", f"paper_{i}")
        pdf_url = paper.get("openAccessPdf", {}).get("url")
        source_url = paper.get("url")

        try:
            # Get full text
            if pdf_url:
                full_text = fetch_pdf_text(pdf_url)
                source_type = "pdf"
            else:
                full_text = f"{title}\n\n{abstract}"
                source_type = "abstract_only"

            chunks = text_splitter.split_text(full_text)

            metadatas = [{
                "title": title,
                "source_type": source_type,
                "source_url": source_url,
                "paper_index": i
            }] * len(chunks)

            ids = [f"{paper_id}_chunk_{j}" for j in range(len(chunks))]

            collection.add(documents=chunks, metadatas=metadatas, ids=ids)
            print(f"✅ Added {len(chunks)} chunks for: {title}")


        except Exception as e:
            print(f"❌ Error processing '{title}': {e}")

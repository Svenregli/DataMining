import requests
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import os
import tempfile

# Set up embedding + ChromaDB
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection("semantic_scholar", embedding_function=openai_ef)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 1. Query Semantic Scholar
def query_semantic_scholar(query: str, limit: int = 10):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,openAccessPdf,url,externalIds,paperId"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("data", [])

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

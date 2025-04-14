import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load env variables (make sure .env has OPENAI_API_KEY)
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load sentence-transformer model
embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Connect to ChromaDB server
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection("paper_chunks")

def search_chunks(query: str, k=6):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    if results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return list(zip(docs, metas))  # ✅ returns [(chunk, metadata), ...]
    else:
        return []




def extract_variables_from_chunks(chunks: list[str]) -> str:
    if not chunks:
        return "No relevant chunks found. Try a more specific query."

    context = "\n\n".join(chunks)
    prompt = f"""You are an academic assistant.

Given these excerpts from a research paper, identify the independent and dependent variables.

### Chunks:
{context}

### Please respond in this format:
Independent Variables: ...
Dependent Variables: ...
"""

    response = openai_client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful academic assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def summarize_chunks(chunks: list[str]) -> str:
    context = "\n\n".join(chunks)
    prompt = f"""You are a research assistant. Read the following excerpts and provide a concise summary of the paper in bullet points.

### Chunks:
{context}

### Summary:"""

    response = openai_client.chat.completions.create(
        model="gpt-4",  # or gpt-3.5-turbo
        messages=[
            {"role": "system", "content": "You are a helpful academic assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

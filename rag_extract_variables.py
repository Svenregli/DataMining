import chromadb
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
import os
import time # Import time for potential backoff
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
# Load env variables (make sure .env has OPENAI_API_KEY)
load_dotenv()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load sentence-transformer model
print("Loading embedding model...")
openai_ef = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
print("Embedding model loaded.")

# Connect to ChromaDB server
try:
    print("Connecting to ChromaDB...")
    client = chromadb.HttpClient(host="localhost", port=8000)
    client.heartbeat()
    collection = client.get_or_create_collection("paper_chunks", embedding_function=openai_ef)
    print("Connected to ChromaDB and collection obtained.")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    collection = None


def search_chunks(query: str, k=6, year=None, author=None, collection_name="arxiv"):
    """
    Searches for relevant text chunks in ChromaDB based on a query and filters.
    """
    if not collection:
        print("ChromaDB collection is not available.")
        return []

    try:
        # Get or create collection based on input name
        collection = client.get_or_create_collection(collection_name)

        results = collection.query(query_texts=[query], n_results=50)

        if not results or not results.get("documents") or not results["documents"][0]:
            print("No results found in ChromaDB for the query.")
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        filtered = []
        for doc, meta in zip(docs, metas):
            meta_year = meta.get("year")
            meta_authors_str = meta.get("authors", "").lower()

            year_match = (year is None or (meta_year is not None and str(meta_year) == str(year)))
            author_match = (author is None or author.strip() == "" or author in meta_authors_str)

            if year_match and author_match:
                filtered.append((doc, meta))

        return filtered[:k]

    except Exception as e:
        print(f"Error during ChromaDB search: {e}")
        return []



def extract_variables_from_chunks(chunks: list[str]) -> str:
    if not chunks:
        return "No text chunks provided for variable extraction."

    context = "\n\n".join(chunks)

    prompt = f"""You are an academic assistant.

Given these excerpts from a research paper, identify the independent and dependent variables mentioned. If no variables are clearly mentioned, state that.

### Excerpts:
{context}

### Please respond ONLY in this format:
Independent Variables: [List variables or state \"None mentioned\"]
Dependent Variables: [List variables or state \"None mentioned\"]
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant specialized in identifying research variables."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    except openai.RateLimitError as e:
        print(f"OpenAI Rate Limit Error: {e}")
        return "Error: API rate limit exceeded. Please try again later."
    except openai.InvalidRequestError as e:
        print(f"OpenAI Invalid Request Error: {e}")
        if "maximum context length" in str(e):
             return f"Error: The provided text is too long for the model ({e}). Consider processing fewer chunks."
        return f"Error: Invalid request to OpenAI API ({e})."
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return "Error: An issue occurred with the OpenAI API. Please try again."
    except Exception as e:
        print(f"An unexpected error occurred during variable extraction: {e}")
        return "Error: An unexpected issue occurred."


def summarize_chunks(chunks: list[str]) -> str:
    if not chunks:
        return "No text chunks provided for summarization."

    context = "\n\n".join(chunks)

    prompt = f"""You are a research assistant. Read the following excerpts from a research paper and provide a concise summary covering the key findings, methods, and conclusions mentioned in the text. Present the summary in clear bullet points.

### Excerpts:
{context}

### Concise Summary (Bullet Points):"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant skilled at summarizing academic texts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()

    except openai.RateLimitError as e:
        print(f"OpenAI Rate Limit Error: {e}")
        return "Error: API rate limit exceeded. Please try again later."
    except openai.InvalidRequestError as e:
        print(f"OpenAI Invalid Request Error: {e}")
        if "maximum context length" in str(e):
             return f"Error: The provided text is too long for the model ({e}). Consider processing fewer chunks or using iterative summarization."
        return f"Error: Invalid request to OpenAI API ({e})."
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return "Error: An issue occurred with the OpenAI API. Please try again."
    except Exception as e:
        print(f"An unexpected error occurred during summarization: {e}")
        return "Error: An unexpected issue occurred."

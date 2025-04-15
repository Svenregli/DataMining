import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI, RateLimitError, APIError, InvalidRequestError # Import specific errors
from dotenv import load_dotenv
import os
import time # Import time for potential backoff

# Load env variables (make sure .env has OPENAI_API_KEY)
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load sentence-transformer model
print("Loading embedding model...")
embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
print("Embedding model loaded.")

# Connect to ChromaDB server
try:
    print("Connecting to ChromaDB...")
    client = chromadb.HttpClient(host="localhost", port=8000)
    client.heartbeat()
    collection = client.get_or_create_collection("paper_chunks")
    print("Connected to ChromaDB and collection obtained.")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    collection = None


def search_chunks(query: str, k=6, year=None, author=None):
    """
    Searches for relevant text chunks in ChromaDB based on a query and filters.
    """
    if not collection:
        print("ChromaDB collection is not available.")
        return []
    try:
        # Fetch more results initially to allow for effective filtering
        results = collection.query(query_texts=[query], n_results=50)

        if not results or not results.get("documents") or not results["documents"][0]:
             print("No results found in ChromaDB for the query.")
             return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        # Apply filters
        filtered = []
        for doc, meta in zip(docs, metas):
            # Ensure metadata values exist before checking
            meta_year = meta.get("year")
            meta_authors_str = meta.get("authors", "").lower() # Get authors as lowercase string

            # Year matching logic (handle potential type mismatch if year is stored as string)
            year_match = (year is None or (meta_year is not None and str(meta_year) == str(year)))

            # Author matching logic (check if the selected author is a substring)
            author_match = (author is None or author.strip() == "" or author in meta_authors_str)

            if year_match and author_match:
                filtered.append((doc, meta)) # returns tuple of (chunk, metadata)

        # Return only the top k filtered results
        return filtered[:k]

    except Exception as e:
        print(f"Error during ChromaDB search: {e}")
        return []


def extract_variables_from_chunks(chunks: list[str]) -> str:
    """
    Extracts independent and dependent variables from a list of text chunks using an LLM.

    Args:
        chunks: A list of strings, where each string is a text chunk.

    Returns:
        A string containing the extracted variables or an error message.
    """
    if not chunks:
        return "No text chunks provided for variable extraction."

    # === Handling Large Context ===
    # WARNING: Joining all chunks can exceed token limits for large documents.
    # Production systems often need iterative processing:
    # 1. Estimate token count of combined chunks.
    # 2. If too large:
    #    a. Process chunks individually or in small batches that fit the limit.
    #    b. Call the API for each batch.
    #    c. Combine the extracted variables from all batches (handle duplicates).
    # For simplicity *now*, we join them, but use a model with a large context window.
    context = "\n\n".join(chunks)
    # TODO: Implement token counting and iterative processing if needed.

    prompt = f"""You are an academic assistant.

Given these excerpts from a research paper, identify the independent and dependent variables mentioned. If no variables are clearly mentioned, state that.

### Excerpts:
{context}

### Please respond ONLY in this format:
Independent Variables: [List variables or state "None mentioned"]
Dependent Variables: [List variables or state "None mentioned"]
"""
    try:
        response = openai_client.chat.completions.create(
            # --- ADJUST MODEL HERE ---
            # Use a model with a large context window if joining chunks.
            # Use gpt-3.5-turbo if implementing iterative processing and cost is a concern.
            model="gpt-4-turbo", # Changed from gpt-4.1-mini
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant specialized in identifying research variables."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Lower temperature for more deterministic extraction
            max_tokens=300 # Limit output tokens to prevent overly long responses
        )
        return response.choices[0].message.content.strip()

    # --- Basic Error Handling ---
    except RateLimitError as e:
        print(f"OpenAI Rate Limit Error: {e}")
        return "Error: API rate limit exceeded. Please try again later."
    except InvalidRequestError as e:
        print(f"OpenAI Invalid Request Error: {e}")
        # Often due to content filters or exceeding max token length for the *model*
        if "maximum context length" in str(e):
             return f"Error: The provided text is too long for the model ({e}). Consider processing fewer chunks."
        return f"Error: Invalid request to OpenAI API ({e})."
    except APIError as e:
        print(f"OpenAI API Error: {e}")
        return "Error: An issue occurred with the OpenAI API. Please try again."
    except Exception as e:
        print(f"An unexpected error occurred during variable extraction: {e}")
        return "Error: An unexpected issue occurred."


def summarize_chunks(chunks: list[str]) -> str:
    """
    Summarizes a list of text chunks using an LLM.

    Args:
        chunks: A list of strings, where each string is a text chunk.

    Returns:
        A string containing the summary or an error message.
    """
    if not chunks:
        return "No text chunks provided for summarization."

    # === Handling Large Context ===
    # WARNING: Joining all chunks can exceed token limits for large documents.
    # Production systems often need iterative processing (Map-Reduce):
    # 1. Estimate token count of combined chunks.
    # 2. If too large:
    #    a. Group chunks into batches that fit the limit.
    #    b. Call the API to summarize each batch -> intermediate summaries.
    #    c. Combine intermediate summaries. If *still* too large, repeat step b.
    #    d. Generate a final summary from the (combined) intermediate summaries.
    # For simplicity *now*, we join them, but use a model with a large context window.
    context = "\n\n".join(chunks)
    # TODO: Implement token counting and iterative processing (map-reduce) if needed.


    prompt = f"""You are a research assistant. Read the following excerpts from a research paper and provide a concise summary covering the key findings, methods, and conclusions mentioned in the text. Present the summary in clear bullet points.

### Excerpts:
{context}

### Concise Summary (Bullet Points):"""

    try:
        response = openai_client.chat.completions.create(
             # --- ADJUST MODEL HERE ---
            model="gpt-4-turbo", # Changed from gpt-4.1-mini
            messages=[
                {"role": "system", "content": "You are a helpful research assistant skilled at summarizing academic texts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5, # Moderate temperature for summarization
            max_tokens=500 # Allow a reasonable length for the summary
        )
        return response.choices[0].message.content.strip()

    # --- Basic Error Handling ---
    except RateLimitError as e:
        print(f"OpenAI Rate Limit Error: {e}")
        return "Error: API rate limit exceeded. Please try again later."
    except InvalidRequestError as e:
        print(f"OpenAI Invalid Request Error: {e}")
        if "maximum context length" in str(e):
             return f"Error: The provided text is too long for the model ({e}). Consider processing fewer chunks or using iterative summarization."
        return f"Error: Invalid request to OpenAI API ({e})."
    except APIError as e:
        print(f"OpenAI API Error: {e}")
        return "Error: An issue occurred with the OpenAI API. Please try again."
    except Exception as e:
        print(f"An unexpected error occurred during summarization: {e}")
        return "Error: An unexpected issue occurred."



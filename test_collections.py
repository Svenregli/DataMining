import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os

client = chromadb.HttpClient(host="localhost", port=8000)
openai_ef = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))

collection = client.get_or_create_collection("semantic_scholar", embedding_function=openai_ef)
print(collection.count())  # ✅ should be > 0

from chromadb import Client
from chromadb.config import Settings
from chromadb.errors import NotFoundError

client = Client(Settings(persist_directory="chroma"))  # ✅ match the app


try:
    client.delete_collection("semantic_scholar")
    print("🗑️ Deleted ChromaDB collection: semantic_scholar")
except NotFoundError:
    print("⚠️ Collection 'semantic_scholar' does not exist — nothing to delete.")


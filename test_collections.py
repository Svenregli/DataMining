import pkg_resources

packages = [
    "streamlit", "chromadb", "python-dotenv", "openai", "sentence-transformers",
    "PyMuPDF", "tqdm", "pandas", "requests", "arxiv", "langchain"
]

for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f"{pkg}: {version}")
    except Exception as e:
        print(f"{pkg}: not found ({e})")

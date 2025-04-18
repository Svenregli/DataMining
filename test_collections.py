from fetch_semantic import query_semantic_scholar, embed_papers_to_chroma
df = query_semantic_scholar("macroeconomic forecasting", limit=10)

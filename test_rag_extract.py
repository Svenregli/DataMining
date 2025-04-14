from rag_extract_variables import search_chunks, extract_variables_from_chunks

query = "effects of machine learning on decision making"
chunks = search_chunks(query)

print(f"\n✅ Found {len(chunks)} chunks.")
if chunks:
    print("🧠 Extracting variables via LLM...\n")
    print(extract_variables_from_chunks(chunks))

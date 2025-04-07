# test_query.py
from retriever import semantic_search

query = "AI agents in organizations"
results = semantic_search(query)

for r in results:
    print(f"\n🔹 {r['title']}")
    print(f"📄 Abstract snippet: {r['abstract'][:200]}...")
    print(f"📈 IV: {r['independent']} → DV: {r['dependent']}")
    print(f"🔗 PDF: {r['url']}")

import pandas as pd

df = pd.read_parquet("data/arxiv_raw.parquet")
print(f"✅ Loaded {len(df)} papers")
print(df.columns)

# Preview first 2 entries
for i, row in df.head(10).iterrows():
    print(f"\n📄 Paper {i}: {row['title']}")
    print(f"Summary: {row['summary'][:300]}...")

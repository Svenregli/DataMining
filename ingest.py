# fetch_arxiv.py
import arxiv
import pandas as pd
import re
from tqdm import tqdm

def fetch_arxiv_papers(query="cat:cs.CL", max_results=200):
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results)
    papers = list(client.results(search))

    dataset = []
    for result in tqdm(papers, desc="📥 Downloading papers"):
        dataset.append({
            'title': result.title,
            'summary': result.summary,
            'pdf_url': re.sub(r'v\d+$', '', result.pdf_url),
            'authors': [a.name for a in result.authors],
            'published': result.published,
            'category': result.primary_category
        })

    df = pd.DataFrame(dataset)
    df.to_parquet("data/arxiv_raw.parquet", index=False)
    print(" Fetched and saved to data/arxiv_raw.parquet")
if __name__ == "__main__":
    fetch_arxiv_papers("cat:cs.CL", max_results=200)

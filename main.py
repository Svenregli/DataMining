# main.py
import arxiv
import pandas as pd
import re
from dotenv import load_dotenv
import os
import asyncio
from searchagent import variable_extraction_agent

load_dotenv()

def ingest_arxiv_papers(query="cat:cs.CL", max_results=1000):
    print(f"Fetching papers with query: {query}")
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    papers = list(client.results(search))

    dataset = []
    for result in papers:
        clean_url = re.sub(r'v\d+$', '', result.pdf_url)
        dataset.append({
            'title': result.title,
            'summary': result.summary,
            'pdf_url': clean_url,
            'authors': [a.name for a in result.authors],
            'published': result.published,
            'primary_category': result.primary_category
        })
    return pd.DataFrame(dataset)


async def extract_variables(df: pd.DataFrame):
    variables = []
    for _, row in df.iterrows():
        prompt = f"""
You are an academic assistant that extracts variables from research papers.

Your task: Identify the independent and dependent variables of the study in the paper at this URL: {row['pdf_url']}

Return the result in this exact format (use only concise, comma-separated phrases):

Independent Variables: ...
Dependent Variables: ...
"""
        try:
            result = await variable_extraction_agent.run(prompt)
            variables.append(result.data)
        except Exception as e:
            print(f" Error for {row['title']}: {e}")
            variables.append("Error")

    df['variables'] = variables
    df[['independent', 'dependent']] = df['variables'].str.extract(
        r"Independent Variables:\s*(.*?)\s*Dependent Variables:\s*(.*)", expand=True
    )
    return df


def save_data(df: pd.DataFrame, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "arxiv_with_variables.parquet"), index=False)
    df.to_csv(os.path.join(out_dir, "arxiv_with_variables.csv"), index=False)
    print("Saved data to disk.")


if __name__ == "__main__":
    df_raw = ingest_arxiv_papers()
    df_processed = asyncio.run(extract_variables(df_raw))
    save_data(df_processed)
    print(df_processed[['title', 'independent', 'dependent']])

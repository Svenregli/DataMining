import arxiv
import pandas as pd
import re
from dotenv import load_dotenv
import os
import asyncio
import matplotlib.pyplot as plt
from collections import Counter
from searchagent import variable_extraction_agent

# Load environment variables
load_dotenv()

# Create ArXiv search client
client = arxiv.Client()

# Perform the search
search = arxiv.Search(
    query='cat:cs.CL',
    max_results=10,  # Keep small for testing
    sort_by=arxiv.SortCriterion.SubmittedDate
)

# Fetch and format results
papers = list(client.results(search))
dataset = []
for result in papers:
    clean_url = re.sub(r'v\d+$', '', result.pdf_url)
    dataset.append({
        'title': result.title,
        'summary': result.summary,
        'pdf_url': clean_url,
        'authors': result.authors,
        'published': result.published,
        'primary_category': result.primary_category
    })

df = pd.DataFrame(dataset)

# Async function to extract variables
async def extract_variables_for_all(df):
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
    return df

# Main runner
if __name__ == "__main__":
    df = asyncio.run(extract_variables_for_all(df))

    # Optional: Split into columns
    df[['independent', 'dependent']] = df['variables'].str.extract(
        r"Independent Variables:\s*(.*?)\s*Dependent Variables:\s*(.*)", expand=True
    )

    # Save to file
    df.to_excel("arxiv_with_variables.xlsx", index=False)
    df.to_csv("arxiv_with_variables.csv", index=False)

    # Print preview
    print(df[['title', 'independent', 'dependent']])


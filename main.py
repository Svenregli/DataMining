import arxiv
import pandas as pd
import re
from dotenv import load_dotenv
import os
import asyncio
from searchagent import variable_extraction_agent  # Also add summary_agent, reference_extraction_agent if needed

# Load environment variables
load_dotenv()

# Create ArXiv search client
client = arxiv.Client()

# Perform the search
search = arxiv.Search(
    query='cat:cs.CL',
    max_results=10,  # Start small for testing
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
            variables.append(result.data)  #  This is the output you want
        except Exception as e:
            print(f" Error for {row['title']}: {e}")
            variables.append("Error")

    df['variables'] = variables
    return df





if __name__ == "__main__":
    df = asyncio.run(extract_variables_for_all(df))
    print(df[['title', 'variables']])
    df.to_csv("arxiv_with_variables.csv", index=False)

import arxiv
import pandas as pd
import re
from dotenv import load_dotenv
import os
from searchagent import summary_agent, reference_extraction_agent,variable_agent

# Load environment variables from .env file
load_dotenv()

# Access the API keys from environment variables
API_KEY = os.getenv('OpenAI_API_KEY')
LLM_MODEL = os.getenv('LLM_MODEL')
LogFire_Key = os.getenv('LogFire_Key')

# Debugging: Print the API keys to verify they are loaded
#print(f"API_KEY: {API_KEY}")
#print(f"LLM_MODEL: {LLM_MODEL}")
#print(f"LogFire_Key: {LogFire_Key}")

# Create an arxiv search client
client = arxiv.Client()

# Perform the search
search = arxiv.Search(
    query='cat:cs.CL',
    max_results=100,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

# Fetch the search results
papers = list(client.results(search))

# Process the search results
dataset = []
for result in papers:
    result.pdf_url = re.sub(r'v\d+$', '', result.pdf_url)
    dataset.append({
        'authors': result.authors,
        'categories': result.categories,
        'entry_id': result.entry_id,
        'pdf_url': result.pdf_url,
        'primary_category': result.primary_category,
        'summary': result.summary,
        'title': result.title,
        'published': result.published,
        'doi': result.doi
    })

# Convert the dataset to a DataFrame
dataset = pd.DataFrame(dataset)



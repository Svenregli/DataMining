import arxiv
import pandas as pd
import re
from config import API_KEY, ANOTHER_API_KEY
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API keys from environment variables
API_KEY = os.getenv('OpenAI_API_KEY')
ANOTHER_API_KEY = os.getenv('ANOTHER_API_KEY')

# Debugging: Print the API keys to verify they are loaded
print(f"API_KEY: {API_KEY}")
print(f"ANOTHER_API_KEY: {ANOTHER_API_KEY}")

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

# Print the first few rows of the DataFrame
print(dataset.head())
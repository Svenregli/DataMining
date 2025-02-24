import arxiv
import pandas as pd
import re

search = arxiv.Search(
    query='cat:cs.CL',
    max_results=1_100,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

client = arxiv.Client()
papers = list(client.results(search))

dataset = []
for result in papers: 
    result.pdf_url = re.sub(r'v\d+$', '', result.pdf_url)
    dataset.append({
        'authors': result.authors,
        'categories': result.categories,
        'entry_id': result.entry_id,
        'pdf_url': result.pdf_url,
        'primary_category': result.primary_category,  # corrected key spelling
        'summary': result.summary,
        'title': result.title,
        'published': result.published,
        'doi': result.doi   
    })

dataset = pd.DataFrame(dataset)
print(dataset.head())

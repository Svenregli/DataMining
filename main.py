import arxiv

search = arxiv.Search(
    query='cat:cs.CL',
    max_results=1_100,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

client = arxiv.Client()
papers = list(client.results(search))
print(papers[1])

{
    'authors': papers[1].authors,
    'categories':papers[1].categories,
    'entry_id': papers[1].entry_id,
    'pdf_url': papers[1].pdf_url,
    'primary_catgeory': papers[1].primary_category,
    'summary': papers[1].summary,
    'title': papers[1].title,
    'published': papers[1].published,
    'doi': papers[1].doi

    
}
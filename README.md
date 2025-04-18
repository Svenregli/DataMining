# 📚 Semantic Scholar Research Assistant

A powerful academic assistant that helps you fetch, embed, analyze, and visualize scientific papers using Semantic Scholar and OpenAI. Built with Streamlit.
You can test it here: https://datamining-dkm9iake685izsyfddtr3q.streamlit.app/

🔁 Step-by-Step Workflow
1. Search & Fetch (Tab 1)
The user enters a topic (e.g., "causal inference in medical AI").

The app queries the Semantic Scholar API to fetch matching papers.

Optionally, the user can fetch the full list of references for each paper.

The papers are displayed, and the user can choose to extract variables or summarize abstracts using GPT.

2. Embed with OpenAI (Tab 1)
When “Fetch and Embed” is selected, each paper’s title and abstract are chunked using RecursiveCharacterTextSplitter.

These chunks are embedded into OpenAI’s vector space using text-embedding-3-small.
On Tab 2 , the user can upload  individual PDF's and summarize them or retrieve the variables, however these uploaded PDF files are not added to the graph (yet)

All embedded chunks are stored in data/vector_store.json.
Be aware that the semantic scholar API has a request limit of 100 request per 5 minutes, therefore only fetch a few papers per query

3. Semantic Retrieval (Tab 3)
The user inputs a search query (e.g., “factors influencing dropout”).

The system embeds the query and performs cosine similarity search against all stored chunks.

Top-k chunks are returned, and the user can:

Extract independent/dependent variables using GPT

Generate a summary of the retrieved content

4. Reference Enrichment (Optional in Tab 4)
Each paper’s references can be enriched by querying the Semantic Scholar API again.

These reference relationships are cached locally in semantic_scholar_cache/.

5. Citation Graph Visualization (Tab 4)
The app builds a directed citation graph using networkx and displays it using PyVis.

Nodes are colored based on the original query file.

Hovering shows titles and metadata (e.g., year).

Physics settings have been tuned for clarity and spacing.
---

## 🚀 Features

- 🔍 **Semantic Scholar Integration** — Search for papers by topic
- 📄 **Chunk & Embed with OpenAI** — Embed paper text into OpenAI’s vector space
- 🧠 **Variable Extraction & Summarization** — Use GPT to analyze retrieved chunks
- 🌐 **Citation Graph Network** — Visualize references and citations with PyVis
- 🧠 **Query-aware vector search** — Retrieve relevant chunks using OpenAI cosine similarity

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/semantic-scholar-assistant.git
cd semantic-scholar-assistant
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Add your OpenAI API key to a `.env` file:

```env
OPENAI_API_KEY=your_openai_key_here
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
├── app.py                            # Main Streamlit app
├── fetch_semantic.py                # Semantic Scholar query + embedding
├── enrich_chroma_papers_with_references.py  # Reference enrichment from API
├── embed_papers_openai.py          # Vector store logic using OpenAI
├── rag_extract_variables.py        # GPT-based extraction and summarization
├── data/
│   ├── vector_store.json            # Embedded chunks
│   ├── semantic_scholar_queries/   # Fetched queries
│   └── semantic_scholar_cache/     # Cached references
├── requirements.txt
└── README.md
```

---

## ✨ Future Ideas

- Add filters by author, year, venue
- Support ArXiv + PDF upload pipelines
- Highlight highly cited nodes
- run project on docker 
-  external storage of vectors, for example in Azure storage accounts to ensure scalability

---

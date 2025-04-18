# 📚 Semantic Scholar Research Assistant

An academic assistant that helps you fetch, embed, analyze, and visualize scientific papers using Semantic Scholar and OpenAI. Built with Streamlit.

👉 **Try it live**: [https://datamining-dkm9iake685izsyfddtr3q.streamlit.app/](https://datamining-dkm9iake685izsyfddtr3q.streamlit.app/)

💸 **Note**: There are around **$10 in OpenAI credit** remaining on the API key. This should be sufficient to analyze ~500–1000 papers depending on their length.
#### 🔍 Why use Semantic Scholar API instead of ArXiv or PubMed?

**✅ Unified Access to Multiple Sources**  
Semantic Scholar aggregates papers from ArXiv, PubMed, Springer, Elsevier, and more — saving you the trouble of querying multiple APIs.  
It currently provides access to **over 215 million papers** (as of April 18, 2025).

**📊 Enriched Metadata Features**  
Semantic Scholar provides structured metadata, including:
- Citations and references
- Paper influence score
- Fields of study
- Author affiliations
- Open access status
- Optionally fetch full reference lists for each paper

---

## 🔁 Step-by-Step Workflow

### 1️⃣ Search & Fetch (Tab 1)

- Enter a topic (e.g., _“causal inference in medical AI”_)
- The app queries the **Semantic Scholar API** to fetch relevant papers
- The results are displayed, and you can:
  - 🧠 Extract independent/dependent variables using GPT
  - 📝 Summarize the abstract
---
### 2️⃣ Embed with OpenAI (Tab 1)

- When “Fetch and Embed” is selected:
  - Each paper’s title and abstract are chunked using `RecursiveCharacterTextSplitter`
  - Chunks are embedded using OpenAI's `text-embedding-3-small`
- Chunks are saved to:  
  📁 `data/vector_store.json`

> ⚠️ **API Rate Limit**: Semantic Scholar allows 100 requests per 5 minutes.  
> 📌 It's best to embed papers **without references first**, then enrich them later.

#### Enrich later via Tab 4:

![image](https://github.com/user-attachments/assets/3693ddda-ac14-417a-a2d7-d6eb8fcb4157)

Tab 4 will:
- Query all stored papers
- Enrich references (if not already present)
- Apply backoff + retry, but stops after 5 failed attempts

---

### 3️⃣ Semantic Retrieval (Tab 3)

- Enter a search query (e.g., _“causal inference in medical AI”_)
- The app embeds the query and performs **cosine similarity** search over all embedded chunks
- You can:
  - 🧠 Extract variables
  - 📄 Summarize results using GPT

---

### 4️⃣ Reference Enrichment (Tab 4)

- Re-fetch full reference metadata from Semantic Scholar
- Results are cached in:  
  📁 `data/semantic_scholar_cache/`

---

### 5️⃣ Citation Graph Visualization (Tab 4)

- A directed graph is built with `networkx` and rendered with `PyVis`
- 🖍 Nodes are **color-coded by query**
- 🪧 Hovering reveals title, year, and metadata

**Central nodes** (papers currently stored and embedded) appear in the middle of the graph:

![image](https://github.com/user-attachments/assets/6be2243f-5ec1-4ec1-bb96-0a35159d61f0)

---
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
- Add a filter to get only publily available papers
- Support ArXiv + PDF upload pipelines
- Highlight highly cited nodes
- run project on docker 
-  external storage of vectors, for example in Azure storage accounts to ensure scalability

---

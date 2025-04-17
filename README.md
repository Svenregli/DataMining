# 🧠 AI-Powered Academic Research Assistant

A local-first, AI-powered assistant that helps you **search**, **analyze**, and **extract structured insights** from academic papers—sourced from both **ArXiv** and **Semantic Scholar**.

Built with OpenAI’s GPT models and **ChromaDB 1.0+**, it enables powerful **semantic search** and variable extraction across chunked paper abstracts.

---

## 🚀 Features

- 🔍 Fetch abstracts from [ArXiv](https://arxiv.org/) and [Semantic Scholar](https://www.semanticscholar.org/)
- ✂️ Chunk content using a sliding window approach
- 🧠 Embed text with `sentence-transformers`
- 💾 Store and query vectors locally with **ChromaDB**
- 💬 Extract insights using OpenAI's GPT models:
  - ✅ Independent & Dependent Variables
  - ✅ Concise Paper Summaries
- 🖥️ Interactive **Streamlit UI**
- 🎛️ Adjustable filters:
  - Source (ArXiv or Semantic Scholar)
  - Category / Keywords
  - Publication year
  - Chunk count
  - Task type (summarization or variable extraction)
- 📦 Persistent vector storage in `./chroma_store/`

---

## 🛠️ Setup Instructions

### 1. Clone and create virtual environment

```bash
git clone https://github.com/Svenregli/DataMining
cd DataMining
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

## 🛠️ Setup Instructions
pip install -r requirements.txt

## 3. Add your OpenAI API key to an env file 
The key has to be in this format: "OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


## 4. Start ChromaDB in a seperate terminal
chroma run --path ./chroma_store

## 5. Launch streamlit App
streamlit run app.py
✨ Example Use Cases
Identify dependent and independent variables in research

Generate quick summaries of papers

Perform semantic similarity search on abstract content

Build structured datasets from unstructured literature

💡 What’s Next?
📄 Support for full PDF ingestion (with Grobid or PDF parsers)

🧠 Visualization of variable networks and research clusters

📊 More filters (e.g., citations, journal, author)

🔁 Cross-source deduplication & similarity linking

🧪 Tech Stack
OpenAI API

ChromaDB 1.0+

Sentence-Transformers

Streamlit

ArXiv API

Semantic Scholar API
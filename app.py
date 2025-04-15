import streamlit as st
from rag_extract_variables import search_chunks, extract_variables_from_chunks, summarize_chunks
from pdf_utils import extract_text_from_pdf
import pandas as pd

# --- Chunk helper ---
def chunk_text(text, size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start += size - overlap
        if start >= len(text):
            break
    if end < len(text) and start < len(text):
        chunks.append(text[start:])
    return chunks

# --- Streamlit App Setup ---
st.set_page_config(
    page_title="🎓 Research Variable Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Academic Assistant")

# --- PDF Upload Section ---
st.markdown("---")
st.markdown("### 📎 Upload Your Own PDF")
pdf_file = st.file_uploader("Upload a PDF file to analyze", type=["pdf"])

if pdf_file:
    with st.spinner(f"Extracting text from {pdf_file.name}..."):
        pdf_text = extract_text_from_pdf(pdf_file)

    user_chunks = chunk_text(pdf_text)
    st.success(f"Extracted {len(user_chunks)} chunks from uploaded PDF.")

    if st.checkbox("Show first chunk of uploaded PDF"):
        if user_chunks:
            st.text_area("First Chunk Preview", user_chunks[0][:1000], height=150)
        else:
            st.warning("No chunks could be generated from the PDF.")

    pdf_task = st.radio(
        "Choose task for the uploaded PDF:",
        ["Extract Variables", "Summarize Paper"],
        key="pdf_task_selector",
        horizontal=True
    )

    if pdf_task == "Extract Variables":
        with st.spinner("🧠 Extracting variables from uploaded PDF..."):
            result = extract_variables_from_chunks(user_chunks)
        st.markdown("#### 🧠 Extracted Variables (from PDF):")
        st.code(result)

    elif pdf_task == "Summarize Paper":
        with st.spinner("📝 Generating summary from uploaded PDF..."):
            result = summarize_chunks(user_chunks)
        st.markdown("#### ✨ Summary (from PDF):")
        st.markdown(result)

# --- ArXiv Search Section ---
st.markdown("---")
st.markdown("### 📚 Or Search ArXiv Papers")

mock_data = {
    'published': pd.to_datetime(['2023-01-15', '2022-05-20', '2023-08-10']),
    'authors': [['Author A', 'Author B'], ['Author C'], ['Author A', 'Author D']],
    'categories': ['cs.LG', 'cs.AI', 'cs.CV']
}
df_meta = pd.DataFrame(mock_data)
df_meta["year"] = pd.to_datetime(df_meta["published"]).dt.year
available_years = sorted(df_meta["year"].unique())

all_authors = df_meta["authors"].explode().dropna().astype(str).unique().tolist()
all_authors.sort()

k = st.slider("How many chunks to retrieve from ChromaDB?", min_value=3, max_value=15, value=6, key="k_slider")

arxiv_task = st.radio(
    "Choose LLM task for ArXiv search results:",
    ["Extract Variables", "Summarize Paper"],
    key="arxiv_task_selector",
    horizontal=True
)

st.sidebar.header("📂 ArXiv Filter Options")
selected_year = st.sidebar.selectbox("Filter by Year", options=["All"] + list(available_years), index=0, key="year_filter")
selected_author = st.sidebar.text_input("Filter by Author Name (optional)", key="author_filter").strip().lower()

query = st.text_input("Enter your research query for ArXiv:", key="query_input")

if query:
    st.markdown("---")
    with st.spinner("🔎 Searching ChromaDB for ArXiv papers..."):
        chunks = search_chunks(
            query,
            k=k,
            year=None if selected_year == "All" else int(selected_year),
            author=selected_author if selected_author else None
        )

    if not chunks:
        st.warning("⚠️ No matching ArXiv chunks found. Try a different query or adjust filters.")
    else:
        st.markdown("### 📄 Retrieved Text Chunks from ArXiv:")
        chunk_texts = []
        for i, (chunk_text, meta) in enumerate(chunks):
            st.markdown(f"#### 🧩 Chunk {i+1} from **{meta.get('title', 'Unknown Title')}**")
            st.markdown(f"[🔗 View paper PDF]({meta.get('url', '#')})")
            st.text_area(f"Chunk {i+1} Preview", f"{chunk_text[:300]}...", height=100, key=f"chunk_preview_{i}")
            chunk_texts.append(chunk_text)

        if arxiv_task == "Extract Variables":
            with st.spinner("🧠 Extracting variables from ArXiv chunks..."):
                result = extract_variables_from_chunks(chunk_texts)
            st.markdown("#### 🧠 Extracted Variables (from ArXiv):")
            st.code(result)

        elif arxiv_task == "Summarize Paper":
            with st.spinner("📝 Generating summary from ArXiv chunks..."):
                result = summarize_chunks(chunk_texts)
            st.markdown("#### ✨ Summary (from ArXiv):")
            st.markdown(result)

st.markdown("---")
st.caption("Academic Assistant Tool")

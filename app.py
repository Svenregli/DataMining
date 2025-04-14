import streamlit as st
from rag_extract_variables import search_chunks, extract_variables_from_chunks
import pandas as pd

st.set_page_config(
    page_title="🎓 Research Variable Extractor",
    layout="wide",
    initial_sidebar_state="expanded"  # ✅ opens sidebar on page load
)
st.title("🔍 Academic Assistant – Variable Extraction")

# Load raw metadata to power filters
df_meta = pd.read_parquet("data/arxiv_raw.parquet")

# Extract available years
df_meta["year"] = pd.to_datetime(df_meta["published"]).dt.year
available_years = sorted(df_meta["year"].unique())

# Create author list (flattened from all papers)
all_authors = df_meta["authors"].explode().dropna().astype(str).unique().tolist()
all_authors.sort()


# 🧠 ArXiv category dropdown
arxiv_categories = {
    "Machine Learning (cs.LG)": "cs.LG",
    "Computation and Language (cs.CL)": "cs.CL",
    "AI (cs.AI)": "cs.AI",
    "Computer Vision (cs.CV)": "cs.CV",
    "Other": "other"
}
selected_category = st.selectbox("Select ArXiv Category", options=list(arxiv_categories.keys()))

# 🔧 Chunk count slider
k = st.slider("How many chunks to retrieve from ChromaDB?", min_value=3, max_value=15, value=6)

# 🎛️ Select what to do with the retrieved chunks
task = st.radio(
    "Choose LLM task:",
    ["Extract Variables", "Summarize Paper"],
    horizontal=True
)


st.sidebar.header("📂 Filter Options")

# Year filter
selected_year = st.sidebar.selectbox("Filter by Year", options=["All"] + list(available_years), index=0)

# Author filter (text input or multiselect)
selected_author = st.sidebar.text_input("Filter by Author Name (optional)").strip().lower()


# 🧠 Query input
query = st.text_input("Enter your research query:")

if query:
    with st.spinner("🔎 Searching ChromaDB..."):
        chunks = search_chunks(
            query,
            k=k,
            year=None if selected_year == "All" else int(selected_year),
            author=selected_author if selected_author else None
        )
    if not chunks:
        st.warning("⚠️ No matching chunks found. Try a broader or more topic-specific query.")
        st.stop()

    st.markdown("### 📄 Retrieved Text Chunks:")

    for i, (chunk_text, meta) in enumerate(chunks):
        st.markdown(f"#### 🧩 Chunk {i+1} from **{meta.get('title', 'Unknown Title')}**")
        st.markdown(f"[🔗 View paper PDF]({meta.get('url', '#')})")
        st.markdown(f"{chunk_text[:300]}...")

chunk_texts = [chunk for chunk, _ in chunks]

if task == "Extract Variables":
    with st.spinner("🧠 Extracting variables..."):
        result = extract_variables_from_chunks(chunk_texts)
    st.markdown("### 🧠 Extracted Variables:")
    st.code(result)

elif task == "Summarize Paper":
    with st.spinner("📝 Generating summary..."):
        from rag_extract_variables import summarize_chunks
        result = summarize_chunks(chunk_texts)
    st.markdown("### ✨ Summary:")
    st.markdown(result)


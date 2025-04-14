import streamlit as st
from rag_extract_variables import search_chunks, extract_variables_from_chunks

st.set_page_config(page_title="🎓 Research Variable Extractor", layout="wide")
st.title("🔍 Academic Assistant – Variable Extraction")

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


# 🧠 Query input
query = st.text_input("Enter your research query:")

if query:
    with st.spinner("🔎 Searching ChromaDB..."):
        chunks = search_chunks(query, k=k)

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


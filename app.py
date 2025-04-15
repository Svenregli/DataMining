import streamlit as st
# Assuming these imports exist and work correctly
# from rag_extract_variables import search_chunks, extract_variables_from_chunks, summarize_chunks
# import pandas as pd
# from pdf_utils import extract_text_from_pdf
# from sentence_transformers import SentenceTransformer
# import chromadb

# Mock functions/data for demonstration if imports fail
import pandas as pd
import time

def search_chunks(query, k, year=None, author=None):
    """ Mock function for searching chunks. """
    print(f"Searching for: {query}, k={k}, year={year}, author={author}")
    # Simulate finding chunks based on query content
    if "findme" in query.lower():
        return [
            (f"This is chunk 1 related to {query}.", {"title": "Paper A", "url": "#"}),
            (f"Another chunk (2) about {query}.", {"title": "Paper B", "url": "#"}),
        ]
    else:
        return [] # Simulate not finding chunks

def extract_variables_from_chunks(chunks):
    """ Mock function for extracting variables. """
    print("Extracting variables...")
    time.sleep(1)
    return f"Extracted variables based on {len(chunks)} chunks:\nVar1, Var2, Var3"

def summarize_chunks(chunks):
    """ Mock function for summarizing chunks. """
    print("Summarizing chunks...")
    time.sleep(1)
    return f"This is a summary based on the content of {len(chunks)} provided chunks."

def extract_text_from_pdf(uploaded_file):
    """ Mock function for PDF text extraction. """
    print(f"Extracting text from {uploaded_file.name}...")
    time.sleep(1)
    # Simulate extracted text
    return "This is simulated text extracted from the PDF. " * 500

def chunk_text(text, size=1000, overlap=200):
    """ Helper function to chunk text. """
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start += size - overlap
        if start >= len(text): # Ensure loop terminates if overlap is large
             break
    # Ensure the last part is included if overlap logic skips it
    if end < len(text) and start < len(text):
         chunks.append(text[start:])
    return chunks

# --- Streamlit App ---

st.set_page_config(
    page_title="🎓 Research Variable Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Academic Assistant")

# --- PDF Upload Section (Moved Up) ---
st.markdown("---") # Add a visual separator
st.markdown("### 📎 Upload Your Own PDF")
pdf_file = st.file_uploader("Upload a PDF file to analyze", type=["pdf"])

pdf_processed = False # Flag to track if PDF processing happened
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

    # Allow choosing task for the uploaded PDF
    pdf_task = st.radio(
        "Choose task for the uploaded PDF:",
        ["Extract Variables", "Summarize Paper"],
        key="pdf_task_selector", # Use a unique key
        horizontal=True
    )

    if pdf_task == "Extract Variables":
        with st.spinner("🧠 Extracting variables from uploaded PDF..."):
            result = extract_variables_from_chunks(user_chunks)
        st.markdown("#### 🧠 Extracted Variables (from PDF):")
        st.code(result)
        pdf_processed = True
    elif pdf_task == "Summarize Paper":
        with st.spinner("📝 Generating summary from uploaded PDF..."):
            result = summarize_chunks(user_chunks)
        st.markdown("#### ✨ Summary (from PDF):")
        st.markdown(result)
        pdf_processed = True

st.markdown("---") # Add another visual separator

# --- ArXiv Search Section ---
st.markdown("### 📚 Or Search ArXiv Papers")

# Mock DataFrame for filters (replace with your actual data loading)
mock_data = {
    'published': pd.to_datetime(['2023-01-15', '2022-05-20', '2023-08-10']),
    'authors': [['Author A', 'Author B'], ['Author C'], ['Author A', 'Author D']],
    'categories': ['cs.LG', 'cs.AI', 'cs.CV'] # Example categories
}
df_meta = pd.DataFrame(mock_data)
df_meta["year"] = pd.to_datetime(df_meta["published"]).dt.year
available_years = sorted(df_meta["year"].unique())

# Explode authors carefully, handling potential lists/strings and NaNs
all_authors = df_meta["authors"].explode().dropna().astype(str).unique().tolist()
all_authors.sort()


# Filters and Task Selection for ArXiv Search
# Note: Removed category filter as it wasn't used in search_chunks mock
# arxiv_categories = {
#     "Machine Learning (cs.LG)": "cs.LG",
#     "Computation and Language (cs.CL)": "cs.CL",
#     "AI (cs.AI)": "cs.AI",
#     "Computer Vision (cs.CV)": "cs.CV",
#     "Other": "other"
# }
# selected_category = st.selectbox("Select ArXiv Category", options=list(arxiv_categories.keys()))

k = st.slider("How many chunks to retrieve from ChromaDB?", min_value=3, max_value=15, value=6, key="k_slider")

arxiv_task = st.radio(
    "Choose LLM task for ArXiv search results:",
    ["Extract Variables", "Summarize Paper"],
    key="arxiv_task_selector", # Use a unique key
    horizontal=True
)

st.sidebar.header("📂 ArXiv Filter Options")
selected_year = st.sidebar.selectbox("Filter by Year", options=["All"] + list(available_years), index=0, key="year_filter")
selected_author = st.sidebar.text_input("Filter by Author Name (optional)", key="author_filter").strip().lower()

query = st.text_input("Enter your research query for ArXiv:", key="query_input")

# Only run ArXiv search if a query is provided
if query:
    st.markdown("---") # Separator before results
    with st.spinner("🔎 Searching ChromaDB for ArXiv papers..."):
        # Pass filters to the search function
        chunks = search_chunks(
            query,
            k=k,
            year=None if selected_year == "All" else int(selected_year),
            author=selected_author if selected_author else None
        )

    if not chunks:
        st.warning("⚠️ No matching ArXiv chunks found. Try a different query or adjust filters.")
        # No st.stop() here, allows the rest of the page (like PDF section) to remain
    else:
        st.markdown("### 📄 Retrieved Text Chunks from ArXiv:")
        chunk_texts = [] # Initialize list to store chunk texts
        for i, (chunk_text, meta) in enumerate(chunks):
            st.markdown(f"#### 🧩 Chunk {i+1} from **{meta.get('title', 'Unknown Title')}**")
            st.markdown(f"[🔗 View paper PDF]({meta.get('url', '#')})") # Assuming URL is in metadata
            st.text_area(f"Chunk {i+1} Preview", f"{chunk_text[:300]}...", height=100, key=f"chunk_preview_{i}")
            chunk_texts.append(chunk_text) # Collect the text of the chunk

        # Perform the selected task on the retrieved ArXiv chunks
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

# Add a footer or some final message if desired
st.markdown("---")
st.caption("Academic Assistant Tool")



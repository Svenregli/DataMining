import streamlit as st
from rag_extract_variables import search_chunks, extract_variables_from_chunks, summarize_chunks
from pdf_utils import extract_text_from_pdf, chunk_text
import pandas as pd
from fetch_semantic import query_semantic_scholar, embed_papers_to_chroma
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
import networkx as nx
import json


# --- Streamlit App Setup ---
st.set_page_config(
    page_title="🎓 Research Variable Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Academic Assistant")
tab1, tab2,tab3,tab4 = st.tabs(["Semantic Scholar","📎 Upload PDF", "📚 ArXiv Search" , "Research citation Network"])

with tab1:

    ###############################################################################################
    # --- Semantic Scholar Search Section ---
    ###############################################################################################
    st.markdown("---")
    st.markdown("### 🧠 Search Semantic Scholar Papers")


    from fetch_semantic import query_semantic_scholar, embed_papers_to_chroma

    user_query = st.text_input("Enter a research topic for Semantic Scholar:", value="causal inference in medical AI", key="semantic_query_input")
    paper_limit = st.slider("How many papers to fetch from Semantic Scholar?", 1, 20, 5, key="semantic_limit_slider")
    fetch_refs = st.checkbox("Fetch references for each paper? slower but will create the Plot in the Research citation Network tab", value=False)
    k_semantic = st.slider("How many chunks to retrieve from ChromaDB?", 3, 15, 6, key="k_semantic_slider")

    st.markdown("#### 🔄 Step 1: Fetch and Embed Papers (Optional)")
    if st.button("Fetch and Embed Semantic Scholar Papers"):
        with st.spinner("Querying Semantic Scholar and embedding results..."):
            try:
                df = query_semantic_scholar(user_query, limit=paper_limit, fetch_references=fetch_refs)
                papers = df.to_dict(orient="records")
                embed_papers_to_chroma(papers)
                st.success(f"✅ Embedded {len(papers)} papers from Semantic Scholar.")
            except Exception as e:
                st.error(f"❌ Failed to fetch or embed papers: {e}")

    st.markdown("#### 🎯 Step 2: Analyze Local ChromaDB (already embedded papers)")

    semantic_task = st.radio(
        "Choose task for Semantic Scholar papers:",
        ["Extract Variables", "Summarize Paper"],
        key="semantic_task_selector",
        horizontal=True
    )

    if user_query and semantic_task:
        with st.spinner("🔎 Searching ChromaDB for matching Semantic Scholar chunks..."):
            chunks = search_chunks(
                query=user_query,
                k=k_semantic,
                collection_name="semantic_scholar"
            )

        if not chunks:
            st.warning("⚠️ No matching chunks found. Try a different query.")
        else:
            chunk_texts = []
            seen_titles = set()

            st.markdown("### 📄 Retrieved Chunks from Semantic Scholar:")

            for i, (chunk_text, meta) in enumerate(chunks):
                title = meta.get("title", "Unknown Title")
                source_url = meta.get("source_url", "#")

                if title not in seen_titles:
                    st.markdown(f"#### 📘 {title}")
                    st.markdown(f"[🔗 View Paper]({source_url})")
                    seen_titles.add(title)

                chunk_texts.append(chunk_text)

            if semantic_task == "Extract Variables":
                with st.spinner("🧠 Extracting variables from Semantic Scholar chunks..."):
                    result = extract_variables_from_chunks(chunk_texts)
                st.markdown("#### 🧠 Extracted Variables (from Semantic Scholar):")
                st.code(result)

            elif semantic_task == "Summarize Paper":
                with st.spinner("📝 Generating summary from Semantic Scholar chunks..."):
                    result = summarize_chunks(chunk_texts)
                st.markdown("#### ✨ Summary (from Semantic Scholar):")
                st.markdown(result)



with tab2:
    ###############################################################################################
    #  --- PDF Upload Section ---
    ##############################################################################################
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


##############################################################################################
# --- ArXiv Search Section ---
##############################################################################################
with tab3:
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
                collection_name="paper_chunks",
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


##############################################################################################
# --- Research Network Visualization Section ---
#############################################################################################

with tab4:  # "Research Network" tab
    st.header("🔗 Research Citation Network")

    try:
        # Load cached results
        with open("data/semantic_scholar_results.json", "r") as f:
            papers = json.load(f)

        # Build the citation graph
        G = nx.DiGraph()
        for paper in papers:
            pid = paper.get("paperId")
            title = paper.get("title", "No Title")
            G.add_node(pid, label=title, title=title)
            for ref in paper.get("references", []):
                ref_url = ref.get("url")
                ref_id = ref_url.split("/")[-1] if ref_url else None
                if ref_id:
                    ref_title = ref.get("title", "Unknown")
                    G.add_node(ref_id, label=ref_title, title=ref_title)
                    G.add_edge(pid, ref_id)

        # Build the PyVis interactive graph
        net = Network(height="600px", width="100%", directed=True)
        net.from_nx(G)
        net.repulsion(node_distance=200, spring_length=200)
        net.save_graph("data/research_network.html")

        # Display it in Streamlit
        with open("data/research_network.html", "r", encoding="utf-8") as f:
            graph_html = f.read()
        components.html(graph_html, height=650, scrolling=True)

    except Exception as e:
        st.error(f"⚠️ Could not load or render research network: {e}")##############################################################################################



st.markdown("---")
st.caption("Academic Assistant Tool")
import streamlit as st
from rag_extract_variables import search_chunks, extract_variables_from_chunks, summarize_chunks
from pdf_utils import extract_text_from_pdf, chunk_text
import pandas as pd
from fetch_semantic import query_semantic_scholar, embed_papers_to_openai_wrapper
from embed_papers_openai import embed_papers_to_openai, embed_and_store_chunks, load_all_cached_papers, enrich_chroma_papers_with_references,load_all_cached_papers_with_sources
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pyvis.network import Network
import streamlit.components.v1 as components
import networkx as nx
import json
import os

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# --- Streamlit App Setup ---
st.set_page_config(
    page_title="🎓 Research Variable Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Academic Assistant")
tab1, tab2,tab3,tab4 = st.tabs(["Semantic Scholar","📎 Upload PDF", "📚 Search stored papers" , "Research citation Network"])

with tab1:
            ###############################################################################################
            #  --- Semantic Scholar Search Section ---
            ##############################################################################################
    st.markdown("---")
    st.markdown("### 🔎 Semantic Scholar Search")

    # Inputs
    user_query = st.text_input("Enter a research topic:", value="causal inference in medical AI")
    paper_limit = st.slider("How many papers to fetch?", 1, 20, 5)
    fetch_refs = st.checkbox("Fetch references for citation network? (slower)", value=False)

    # Mode selection
    mode = st.radio("Choose what to do:", [
        "Fetch and Analyze (no embedding)",
        "Fetch and Embed to OpenAI Vector Store"
    ])

    # Process query
    if st.button("🚀 Run Query"):
        with st.spinner("Fetching papers from Semantic Scholar..."):
            try:
                df = query_semantic_scholar(user_query, limit=paper_limit, fetch_references=fetch_refs)
                papers = df.to_dict(orient="records")
                st.success(f"✅ Fetched {len(papers)} papers.")

                if mode == "Fetch and Embed to OpenAI Vector Store":
                    embed_papers_to_openai_wrapper(papers)
                    st.success(f"✅ Embedded {len(papers)} papers using OpenAI embeddings.")

                # Show papers + select
                titles = [p["title"] for p in papers]
                selected_title = st.selectbox("Choose a paper to analyze:", titles)
                selected_paper = next(p for p in papers if p["title"] == selected_title)

                st.markdown(f"### 📘 {selected_paper['title']}")
                st.markdown(f"**Abstract:** {selected_paper.get('abstract', 'No abstract available')}")

                # LLM Task
                task = st.radio("Run on Abstract:", ["Extract Variables", "Summarize Paper"])
                if task == "Extract Variables":
                    with st.spinner("Extracting variables..."):
                        result = extract_variables_from_chunks([selected_paper.get("abstract", "")])
                    st.code(result)
                elif task == "Summarize Paper":
                    with st.spinner("Summarizing..."):
                        result = summarize_chunks([selected_paper.get("abstract", "")])
                    st.markdown(result)

                # Inline citation graph (only if references were fetched)
                if fetch_refs:
                    st.markdown("### 🔗 Citation Network for This Query")
                    G = nx.DiGraph()
                    for paper in papers:
                        pid = paper["paperId"]
                        title = paper["title"]
                        G.add_node(pid, label=title)
                        for ref in paper.get("references", []):
                            ref_id = ref.get("paperId")
                            if ref_id:
                                G.add_node(ref_id, label=ref.get("title", "Unknown"))
                                G.add_edge(pid, ref_id)

                    net = Network(height="500px", width="100%", directed=True)
                    net.from_nx(G)
                    net.save_graph("data/temp_query_network.html")
                    with open("data/temp_query_network.html", "r", encoding="utf-8") as f:
                        html = f.read()
                    components.html(html, height=550, scrolling=True)

            except Exception as e:
                st.error(f"❌ Failed to fetch or process papers: {e}")



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
# --- Retrieval Search Section ---
##############################################################################################
with tab3:
    st.markdown("---")
    st.markdown("### 🧠 Analyze Embedded Papers with OpenAI Vector Search")

    query = st.text_input("Search embedded paper chunks:", value="causal inference in medical AI")
    k = st.slider("How many chunks to retrieve:", 3, 20, 6)

    if query:
        with st.spinner("Searching with OpenAI vector embeddings..."):
            chunks = search_chunks(query=query, k=k)

        if not chunks:
            st.warning("⚠️ No matching chunks found. Try another query.")
        else:
            chunk_texts = []
            seen_titles = set()

            st.markdown("### 📄 Retrieved Chunks:")
            for i, (chunk_text, meta) in enumerate(chunks):
                title = meta.get("title", "Unknown Title")
                source_url = meta.get("source_url", "#")

                if title not in seen_titles:
                    st.markdown(f"#### 📘 {title}")
                    st.markdown(f"[🔗 View Paper]({source_url})")
                    seen_titles.add(title)

                chunk_texts.append(chunk_text)

            task = st.radio("Choose analysis task:", ["Extract Variables", "Summarize Paper"], horizontal=True)

            if task == "Extract Variables":
                with st.spinner("Extracting variables from retrieved chunks..."):
                    result = extract_variables_from_chunks(chunk_texts)
                st.markdown("#### 🧠 Extracted Variables:")
                st.code(result)

            elif task == "Summarize Paper":
                with st.spinner("Summarizing retrieved chunks..."):
                    result = summarize_chunks(chunk_texts)
                st.markdown("#### ✨ Summary:")
                st.markdown(result)


##############################################################################################
# --- Research Network Visualization Section ---
#############################################################################################

# Tab 4: Global Research Citation Network

def load_enriched_papers(queries_folder="data/semantic_scholar_cache", cache_folder="data/semantic_scholar_cache"):
    all_papers = []
    for file in os.listdir(queries_folder):
        if file.endswith(".json"):
            path = os.path.join(queries_folder, file)
            query_name = file.replace("search_", "").replace(".json", "")
            try:
                with open(path, "r") as f:
                    papers = json.load(f)
                    for paper in papers:
                        pid = paper.get("paperId")
                        ref_path = os.path.join(cache_folder, f"paper_{pid}.json")
                        if os.path.exists(ref_path):
                            try:
                                with open(ref_path, "r") as ref_file:
                                    references = json.load(ref_file)
                                paper["references"] = references
                            except:
                                paper["references"] = []
                        else:
                            paper["references"] = []
                        paper["source"] = query_name
                        all_papers.append(paper)
            except Exception as e:
                st.warning(f"Failed to load {file}: {e}")
    return all_papers

# Generate unique colors per query source
def generate_color_palette(sources):
    base_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    palette = {}
    for i, source in enumerate(sorted(set(sources))):
        palette[source] = base_colors[i % len(base_colors)]
    return palette

# Tab 4: Global Research Citation Network
with tab4:  # "Research Network" tab
    st.markdown("---")
    st.markdown("### 🌐 Global Research Citation Network (from cached queries)")

    if st.button("📡 Enrich References for All Embedded Papers"):
        with st.spinner("Fetching references from Semantic Scholar for embedded papers..."):
            count = enrich_chroma_papers_with_references()
            st.success(f"✅ Enriched {count} papers with reference metadata.")

    papers = load_enriched_papers()

    if not papers:
        st.warning("⚠️ No enriched papers found. Please run a Semantic Scholar query and enrich references in Tab 1.")
    else:
        st.success(f"Loaded {len(papers)} enriched papers.")

        G = nx.DiGraph()
        edges = 0

        sources = [p.get("source", "unknown") for p in papers]
        color_map = generate_color_palette(sources)

        for paper in papers:
            pid = paper.get("paperId")
            title = paper.get("title", "Untitled")
            source = paper.get("source", "unknown")
            year = paper.get("year")

            if pid:
                G.add_node(
                    pid,
                    label=title[:60],
                    title=f"{title}\nYear: {year}",
                    color=color_map.get(source, "#999")
                )

                for ref in paper.get("references", []):
                    ref_id = ref.get("paperId")
                    ref_title = ref.get("title", "Unknown")
                    if ref_id:
                        G.add_node(ref_id, label=ref_title[:60], title=ref_title)
                        G.add_edge(pid, ref_id)
                        edges += 1

        st.text(f"📊 Graph has {G.number_of_nodes()} nodes and {edges} edges.")

        if edges == 0:
            st.warning("⚠️ No citation edges were created. Make sure references have 'paperId' fields.")
        else:
            net = Network(
                height="800px",
                width="100%",
                directed=True,
                notebook=False,
                bgcolor="#ffffff",
                font_color="black"
            )
            net.from_nx(G)
            net.set_options("""
var options = {
  "nodes": {
    "font": {
      "size": 18
    },
    "shape": "dot",
    "size": 15
  },
  "edges": {
    "color": {
      "inherit": true
    },
    "smooth": false
  },
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -30000,
      "centralGravity": 0.3,
      "springLength": 400,
      "springConstant": 0.04,
      "damping": 0.09,
      "avoidOverlap": 1
    },
    "minVelocity": 0.75
  }
}
""")


            net.save_graph("data/global_citation_network.html")

            with open("data/global_citation_network.html", "r", encoding="utf-8") as f:
                html = f.read()
            components.html(html, height=850, scrolling=True)

        missing_refs = sum(1 for p in papers for r in p.get("references", []) if not r.get("paperId"))
        st.info(f"🧩 Found {missing_refs} references missing 'paperId'.")





st.markdown("---")
st.caption("Academic Assistant Tool")
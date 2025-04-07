# app.py
import streamlit as st
from retriever import semantic_search

st.set_page_config(page_title="AI Research Finder", layout="wide")

st.title("🔍 Semantic Research Paper Search")

query = st.text_input("Enter a research topic or question:")

if query:
    with st.spinner("Searching..."):
        results = semantic_search(query, k=5)

    st.markdown("### Top Results:")
    for r in results:
        st.markdown(f"#### [{r['title']}]({r['url']})")
        st.markdown(f"- **Independent Variables**: {r['independent']}")
        st.markdown(f"- **Dependent Variables**: {r['dependent']}")
        st.markdown(f"*Abstract*: {r['abstract'][:500]}...")
        st.markdown("---")

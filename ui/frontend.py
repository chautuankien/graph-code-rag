import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import streamlit as st
import zipfile
import os
import shutil

from src.code_graph_rag.agent.graph_agent import graph, GraphState
from src.code_graph_rag.pipeline import build_knowledge_graph_and_insert_db

# Sidebard
st.sidebar.header("📦 Upload Codebase")

uploaded_file = st.sidebar.file_uploader("Upload a zipped Python repo", type="zip")

UPLOAD_DIR = "uploaded_repo"
if uploaded_file is not None:
    # Clear old uploaded folder
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Extract zip into UPLOAD_DIR
    with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
        zip_ref.extractall(UPLOAD_DIR)
    
    # Build graph from uploaded repo
    try:
        st.sidebar.write("🔄 Building graph from uploaded repo...")
        build_knowledge_graph_and_insert_db(UPLOAD_DIR)
        st.sidebar.success("✅ Graph loaded from uploaded repo.")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")

st.title("📘 Codebase RAG Agent")

question = st.text_input("Enter your question about the codebase:")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        state = GraphState(question=question)
        result = graph.invoke(state)
        result = GraphState.model_validate(result)

        st.markdown("### ✅ Answer")
        st.write(result.answer)

        with st.expander("🔍 Cypher Query"):
            st.code(result.cypher_query or "No query generated")

        with st.expander("📚 Context"):
            for snippet in result.code_snippets or []:
                st.code(snippet, language="python")
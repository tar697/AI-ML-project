# app.py
import streamlit as st
import sys
import os

# Add src/ to Python path
current_dir = os.path.dirname(__file__)
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import the RAG functions
try:
    from rag_pipeline import answer_question, load_resources
except Exception as e:
    st.error(f"Failed to import RAG pipeline: {e}")
    st.stop()

# Streamlit config
st.set_page_config(page_title="KCC Query Assistant", layout="wide")
st.title("🚜 KCC Query Assistant")
st.markdown("Ask a question related to farming. The assistant will try to answer based on the KCC dataset or fallback to web search.")

# Load AI model and index only once
@st.cache_resource
def load_all_resources_cached():
    try:
        load_resources()
        return True
    except Exception as e:
        st.error(f"Error loading AI resources: {e}")
        return False

resources_loaded = load_all_resources_cached()

# Streamlit UI
if resources_loaded:
    user_query = st.text_input("Enter your question:", placeholder="e.g., What are the control measures for aphids in cotton?")

    if st.button("Get Answer"):
        if user_query.strip() != "":
            with st.spinner("Generating answer..."):
                try:
                    answer, source = answer_question(user_query)
                    st.subheader("Answer:")
                    st.markdown(answer)

                    if source == "KCC Dataset":
                        st.info("📘 Based on the KCC Dataset.")
                    elif source == "Web Search":
                        st.warning("🌐 Based on web search (DuckDuckGo).")
                    else:
                        st.error("❌ No clear answer found.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question.")
else:
    st.error("Failed to load AI resources. Check terminal for details.")

st.markdown("---")
st.caption("🔍 KCC Query Assistant — Powered by Sentence Transformers, FAISS & Streamlit.")

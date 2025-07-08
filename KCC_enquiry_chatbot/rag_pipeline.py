# rag_pipeline.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
import os

model = None
index = None
questions = None
answers = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "../faiss_index.index")
TEXT_DATA_PATH = os.path.join(BASE_DIR, "../text_data.pkl")

def load_resources():
    global model, index, questions, answers

    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(INDEX_PATH)

    with open(TEXT_DATA_PATH, "rb") as f:
        data = pickle.load(f)
        questions = data['questions']
        answers = data['answers']

def web_search(query, max_results=2):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
        if results:
            return "\n\n".join([f"{r['title']}\n{r['body']}" for r in results])
        return "No relevant results found on the web."

def answer_question(query, top_k=3, distance_threshold=1.0):
    if model is None or index is None:
        raise RuntimeError("Resources not loaded. Call load_resources() first.")

    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    if distances[0][0] > distance_threshold:
        web_result = web_search(query)
        return web_result, "Web Search"

    results = []
    for i in indices[0]:
        if i < len(questions) and i < len(answers):
            q = questions[i]
            a = answers[i]
            results.append(f"**Q:** {q}\n**A:** {a}")

    if results:
        return "\n\n".join(results), "KCC Dataset"
    else:
        return "No answer could be found in the dataset.", "System (No Answer)"

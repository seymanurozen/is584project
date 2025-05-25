import os
import faiss
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report

FAISS_INDEX_PATH = "retriever/faiss.index"
CONTEXT_CSV_PATH = "retriever/context_snippets.csv"
PAPER_CSV_PATH = "asap_reviews_labeled.csv"

TOP_K = 5
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"

embedding_model = SentenceTransformer(EMBED_MODEL, device="cpu")

print("Loading input data...")
df_paper = pd.read_csv(PAPER_CSV_PATH)
df_ctx = pd.read_csv(CONTEXT_CSV_PATH)
ctx_embeddings = embedding_model.encode(df_ctx["span"].tolist(), show_progress_bar=True, device="cpu")

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

def build_prompt(paper_text, retrieved_snippets):
    context = "\n- " + "\n- ".join(retrieved_snippets)
    prompt = (
        "You are an academic reviewer assistant.\n"
        "Paper Abstract:\n" + paper_text + "\n\n"
        "Relevant past peer review excerpts (summary, motivation, clarity, soundness):" + context + "\n\n"
        "Instruction:\n"
        "Determine whether this paper should be accepted.\n"
        "If the reviews are positive and the paper is clear and novel, respond with 'Accept'.\n"
        "If the reviews raise major concerns or the paper lacks clarity or novelty, respond with 'Reject'.\n"
        "Answer with a single word: Accept or Reject."
    )
    return prompt

def predict_acceptance_with_ollama(prompt):
    response = ollama.chat(model=OLLAMA_MODEL, messages=[
        {"role": "user", "content": prompt}
    ])
    answer = response["message"]["content"].strip().lower()
    return 1 if "accept" in answer else 0

print("Starting prediction using Ollama...")
preds = []
true_labels = []

for _, row in tqdm(df_paper.iterrows(), total=len(df_paper)):
    paper_id, paper_text, true_label = row["id"], row["text"], row["label"]
    query_vec = embedding_model.encode([paper_text], device="cpu")
    top_k_scores, top_k_idx = index.search(np.array(query_vec).astype("float32"), TOP_K)
    retrieved_spans = df_ctx.iloc[top_k_idx[0]]['span'].tolist()
    prompt = build_prompt(paper_text, retrieved_spans)
    pred = predict_acceptance_with_ollama(prompt)
    preds.append(pred)
    true_labels.append(true_label)

print("\nClassification Report (Ollama-based RAG):")
print(classification_report(true_labels, preds, digits=4))

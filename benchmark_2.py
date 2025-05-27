import os
import faiss
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, accuracy_score

# Paths
FAISS_INDEX_PATH = "retriever/faiss.index"
CONTEXT_CSV_PATH = "retriever/context_snippets.csv"
PAPER_CSV_PATH = "asap_reviews_labeled.csv"

# Search and generation configs
TOP_K_VALUES = [3, 5, 7]
MAX_TOKEN_VALUES = [512, 768, 1024]
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"

# Load models and data
embedding_model = SentenceTransformer(EMBED_MODEL, device="cpu")
print("Loading data...")
df_paper = pd.read_csv(PAPER_CSV_PATH)
df_ctx = pd.read_csv(CONTEXT_CSV_PATH)
ctx_embeddings = embedding_model.encode(df_ctx["span"].tolist(), show_progress_bar=True, device="cpu")
print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

# Prompt template
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

# Single prediction
def predict_acceptance(prompt, max_tokens):
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 10, "num_ctx": max_tokens}
    )
    answer = response["message"]["content"].strip().lower()
    return 1 if "accept" in answer else 0

# Start sweep
wandb.init(project="is584-benchmark", name="rag-hyperparam-grid")

results = []

for top_k in TOP_K_VALUES:
    for max_tokens in MAX_TOKEN_VALUES:
        print(f"\nRunning TOP_K={top_k}, MAX_TOKENS={max_tokens}...")
        preds = []
        true_labels = []

        for _, row in tqdm(df_paper.iterrows(), total=len(df_paper)):
            paper_text = row["text"]
            true_label = row["label"]
            query_vec = embedding_model.encode([paper_text], device="cpu")
            _, top_k_idx = index.search(np.array(query_vec).astype("float32"), top_k)
            retrieved_spans = df_ctx.iloc[top_k_idx[0]]["span"].tolist()
            prompt = build_prompt(paper_text, retrieved_spans)
            pred = predict_acceptance(prompt, max_tokens)
            preds.append(pred)
            true_labels.append(true_label)

        report = classification_report(true_labels, preds, digits=4, output_dict=True)
        acc = accuracy_score(true_labels, preds)

        wandb.log({
            "top_k": top_k,
            "max_input_tokens": max_tokens,
            "accuracy": acc,
            "f1_macro": report['macro avg']['f1-score'],
            "precision_macro": report['macro avg']['precision'],
            "recall_macro": report['macro avg']['recall']
        })

        results.append({
            "top_k": top_k,
            "max_input_tokens": max_tokens,
            "accuracy": acc,
            "f1_macro": report['macro avg']['f1-score'],
            "precision_macro": report['macro avg']['precision'],
            "recall_macro": report['macro avg']['recall']
        })

print("\nBenchmark Summary:")
for r in results:
    print(f"TOP_K={r['top_k']} | TOKENS={r['max_input_tokens']} | Acc={r['accuracy']:.4f} | F1={r['f1_macro']:.4f} | Prec={r['precision_macro']:.4f} | Rec={r['recall_macro']:.4f}")

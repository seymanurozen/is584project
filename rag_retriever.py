import os
import json
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


ASPECT_WHITELIST = {"summary", "motivation_positive", "clarity_positive", "soundness_positive"}
REVIEW_JSONL = "dataset/dataset/aspect_data/review_with_aspect.jsonl"
OUTPUT_PICKLE = "retriever/faiss_index.pkl"
OUTPUT_CSV = "retriever/context_snippets.csv"

print("Loading embedding model")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Extracting aspect-based spans...")
examples = []
with open(REVIEW_JSONL, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        item = json.loads(line)
        rid = item["id"]
        text = item["text"]
        for start, end, label in item["labels"]:
            if label in ASPECT_WHITELIST:
                span = text[start:end].strip()
                if 20 < len(span) < 512:  # filter out very short or long spans
                    examples.append({"id": rid, "label": label, "span": span})

print(f"Embedding {len(examples)} aspect snippets...")
texts = [e["span"] for e in examples]
embeddings = model.encode(texts, show_progress_bar=True, device='cpu')


# Build FAISS index
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings).astype("float32"))

# Save outputs
os.makedirs("retriever", exist_ok=True)
pd.DataFrame(examples).to_csv(OUTPUT_CSV, index=False)
faiss.write_index(index, "retriever/faiss.index")
print("FAISS index and context snippets saved.")

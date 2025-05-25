
import os
import json
import pandas as pd
from glob import glob

REVIEW_JSONL = "dataset/dataset/aspect_data/review_with_aspect.jsonl"  # <== senin belirttiğin doğru path
PAPER_META_DIRS = [
    "dataset/dataset/ICLR_2017/ICLR_2017_paper",
    "dataset/dataset/ICLR_2018/ICLR_2018_paper",
    "dataset/dataset/ICLR_2019/ICLR_2019_paper",
    "dataset/dataset/ICLR_2020/ICLR_2020_paper",
    "dataset/dataset/NIPS_2016/NIPS_2016_paper",
    "dataset/dataset/NIPS_2017/NIPS_2017_paper",
    "dataset/dataset/NIPS_2018/NIPS_2018_paper",
    "dataset/dataset/NIPS_2019/NIPS_2019_paper"
]


print("Loading reviews...")
reviews = {}
with open(REVIEW_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        reviews[item["id"]] = item["text"]

print("Extracting paper decisions...")
decisions = {}
for folder in PAPER_META_DIRS:
    json_files = glob(os.path.join(folder, "*.json"))
    print(f"📂 {folder}: {len(json_files)} files found")
    for filepath in json_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            paper_id = data.get("id")
            decision = data.get("decision", "").lower()
            if paper_id and ("accept" in decision or "reject" in decision):
                label = 1 if "accept" in decision else 0
                decisions[paper_id] = label


total_reviews = len(reviews)
print("Merging review texts with decisions...")
examples = []
missing_decisions = 0
for pid, text in reviews.items():
    if pid in decisions:
        examples.append({"id": pid, "text": text, "label": decisions[pid]})
    else:
        missing_decisions += 1

print(f"Number of reviews without decision information: {missing_decisions} / {total_reviews}")
print(f"Matched review: {len(examples)}")

# 5. CSV olarak kaydet
df = pd.DataFrame(examples)
OUTPUT_PATH = "asap_reviews_labeled.csv"
df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to: {OUTPUT_PATH}")

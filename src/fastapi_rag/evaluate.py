import json
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path
import sys
from datetime import date

QA_FILE = Path("evals/qa.json")

rows = json.loads(QA_FILE.read_text())
answerable = [r for r in rows if r["answerable"]]

K = 3
#till which rank it will check is called depth
MRR_DEPTH = 10

load_dotenv()
client = genai.Client()

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[r["question"] for r in answerable],
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
)
vectors = [e.values for e in result.embeddings]

collection = chromadb.PersistentClient(path="chroma_db").get_collection("fastapi_docs")
res = collection.query(query_embeddings=vectors, n_results=MRR_DEPTH)

def recall_at_k(retrieved, relevant, k):
    #it will do and and return the set list like which are in both
    found = relevant & set(retrieved[:k])
    return len(found) / len(relevant)

def reciprocal_rank(retrieved, relevant):
    for rank, source in enumerate(retrieved, start=1):
        if source in relevant:
            return 1 / rank
    return 0.0

def contains(chunk, phrase):
    return phrase.lower() in chunk.lower()

def chunk_recall_at_k(chunks, phrases, k):
    found = [p for p in phrases if any(contains(c, p) for c in chunks[:k])]
    return len(found) / len(phrases)

def chunk_reciprocal_rank(chunks, phrases):
    for rank, chunk in enumerate(chunks, start=1):
        if any(contains(chunk, p) for p in phrases):
            return 1 / rank
    return 0.0

per_question = []
for row, metas, docs in zip(answerable, res["metadatas"], res["documents"]):
    retrieved = [m["source"] for m in metas]
    relevant = set(row["sources"])

    recall = recall_at_k(retrieved, relevant, K)
    rr = reciprocal_rank(retrieved, relevant)

    c_recall = chunk_recall_at_k(docs, row["evidence"], K)
    c_rr = chunk_reciprocal_rank(docs, row["evidence"])
    if row["id"] == "q05":
        for i, d in enumerate(docs[:3], start=1):
            print(f"--- rank {i} ---\n{d}\n")

    per_question.append({"id": row["id"], "recall": recall, "rr": rr, "c_recall": c_recall, "c_rr": c_rr, "top3": retrieved[:K]})

n = len(per_question)
recall_avg = sum(q["recall"] for q in per_question) / n
mrr = sum(q["rr"] for q in per_question) / n
c_recall_avg = sum(q["c_recall"] for q in per_question) / n
c_mrr = sum(q["c_rr"] for q in per_question) / n


print(f"{'id':5} {'recall':>6} {'rr':>5} {'c_rec':>6} {'c_rr':>5}  top-{K} sources")
for q in per_question:
    flag = "" if q["c_rr"] == 1 and q["c_recall"] == 1 else "  <--"
    print(f"{q['id']:5} {q['recall']:6.2f} {q['rr']:5.2f} {q['c_recall']:6.2f} {q['c_rr']:5.2f}  {', '.join(q['top3'])}{flag}")

print(f"\nfile level:  recall@{K} = {recall_avg:.3f}   MRR@{MRR_DEPTH} = {mrr:.3f}")
print(f"chunk level: recall@{K} = {c_recall_avg:.3f}   MRR@{MRR_DEPTH} = {c_mrr:.3f}   ({n} questions)")


label = sys.argv[1] if len(sys.argv) > 1 else "run"

RESULTS_DIR = Path("evals/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

out = RESULTS_DIR / f"{label}.json"
out.write_text(
    json.dumps({
        "label": label,
        "date": str(date.today()),
        "config": {"chunk_size": 250, "chunk_overlap": 30, "embed_model": "gemini-embedding-001"},
        "chunks": collection.count(),
        f"recall@{K}": round(recall_avg, 3),
        f"mrr@{MRR_DEPTH}": round(mrr, 3),
        f"chunk_recall@{K}": round(c_recall_avg, 3),
        f"chunk_mrr@{MRR_DEPTH}": round(c_mrr, 3),
        "per_question": per_question,
    }, indent=2)
)

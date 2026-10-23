import json
import time
from pathlib import Path
from google.genai import errors

from fastapi_rag.rag import answer, GEN_MODEL, TOP_K

QA_FILE = Path("evals/qa.json")
OUT_FILE = Path("evals/results/answers.json")

def save(results):
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({"gen_model": GEN_MODEL, "top_k": TOP_K, "answers": results}, indent=2))


rows = json.loads(QA_FILE.read_text())

#Save answer generated and retry for those for which we didnt got the answer
results = json.loads(OUT_FILE.read_text())["answers"] if OUT_FILE.exists() else []
done = {r["id"] for r in results}
failed = []

for row in rows:
    if row["id"] in done:
        continue
    try:
        out = answer(row["question"])
    except errors.APIError as e:
        print(f"{row['id']}: FAILED after retries ({e.code}): {e.message}")
        failed.append(row["id"])
        continue
    out = answer(row["question"])
    results.append({
        "id": row["id"],
        "question": row["question"],
        "answerable": row["answerable"],
        "answer": out["answer"],
        "chunks": out["chunks"],
      }  
    )
    save(results)
    print(f"{row['id']}: {out['answer'][:80]!r}")
    time.sleep(8)

print(f"\n{len(results)}/{len(rows)} answered. Failed: {failed or 'none'}")
if failed:
    print("Run the script again to retry only the failed ones.")

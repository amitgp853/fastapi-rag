import json
import time
from pathlib import Path

from google.genai import errors, types
from pydantic import BaseModel

from fastapi_rag.rag import client

JUDGE_MODEL = "gemini-3.8-flash"
ANSWERS_FILE = Path("evals/results/answers.json")
OUT_FILE = Path("evals/results/faithfulness.json")

JUDGE_PROMPT = """You are grading a RAG system's answer for FAITHFULNESS.

Faithful means every factual claim in the ANSWER is supported by the CONTEXT.
- Judge only against the CONTEXT. Do not use your own knowledge of whether a claim is true.
- A claim the context does not state is unsupported, even if it is correct in real life.
- Paraphrasing and obvious rewording of the context count as supported.

First list each claim in the answer, then list the unsupported ones, then explain, then give the verdict.
faithful = 1 only if every claim is supported, otherwise 0.

CONTEXT:
{context}

ANSWER:
{answer}"""

class Verdict(BaseModel):
    claims: list[str]
    unsupported: list[str]
    reason: str
    faithful: int

def is_idk(text):
    return text.strip().rstrip(".").lower() == "i don't know"

def judge(context, answer):
    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=JUDGE_PROMPT.format(
            context=context, answer=answer
        ),
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=Verdict,
        )
    )
    return response.parsed

def save(results):
    OUT_FILE.write_text(json.dumps({"judge_model": JUDGE_MODEL, "verdicts": results}, indent=2))

answers = json.loads(ANSWERS_FILE.read_text())["answers"]
results = json.loads(OUT_FILE.read_text())["verdicts"] if OUT_FILE.exists() else []
done = {r["id"] for r in results}
failed = []

for a in answers:
    if a["id"] in done:
        continue
    if is_idk(a["answer"]):
        results.append({"id": a["id"], "abstained": True})
        save(results)
        continue

    context = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in a["chunks"])

    try:
        verdict = judge(context, a["answer"])
    except errors.APIError as e:
        print(f"{a['id']}: FAILED ({e.code}): {e.message}")
        failed.append(a["id"])
        continue
    
    if verdict is None or verdict.faithful not in (0, 1):
        print(f"{a['id']}: FAILED (judge output invalid)")
        failed.append(a["id"])
        if e.code == 429:
            print("Quota exhausted. Stopping; rerun later to resume.")
            break
        continue

    results.append({"id": a["id"],  "abstained": False, **verdict.model_dump()})
    save(results)
    print(f"{a['id']}: faithful={verdict.faithful} {verdict.reason[:90]}")
    time.sleep(4)

judged = [r for r in results if not r["abstained"]]
if not judged:
    print("Nothing judged. Check the failures above.")
else:
    score = sum(r["faithful"] for r in judged)/len(judged)
    print(f"\nfaithfulness = {score:.3f}  ({len(judged)} judged, {len(results) - len(judged)} abstained)")
print(f"Failed: {failed or 'none'}")
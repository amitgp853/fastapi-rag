import json
from pathlib import Path

from fastapi_rag.judge import judge

TESTS_FILE = Path("evals/judge_tests.json")

tests = json.loads(TESTS_FILE.read_text())
passed = 0

for t in tests:
    context = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in t["chunks"])
    verdict = judge(context, t["answer"])

    ok = verdict is not None and verdict.faithful == t["expected"]
    passed += ok
    got = verdict.faithful if verdict else "invalid"
    print(f"{t['id']} {t['case']:25} expected={t['expected']} got={got}  {'PASS' if ok else 'FAIL'}")
    if verdict:
        print(f"    unsupported: {verdict.unsupported}")
        print(f"    reason: {verdict.reason[:150]}")

print(f"\n{passed}/{len(tests)} passed")

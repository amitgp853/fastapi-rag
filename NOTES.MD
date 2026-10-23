###### WEEK 4 ########

======= Day 1 ========

The big idea: a RAG app can fail in two places, so you measure both.

Question ─► RETRIEVAL (did we fetch the right chunks?) ─► GENERATION (did the LLM answer correctly from them?) ─► Answer
measured by retrieval metrics                  
measured by generation metrics

Hit rate @k: did any correct chunk appear in the top k?

Recall @k: what fraction of the relevant chunks appeared in the top k?
If a question needs 2 chunks and you retrieved 1 of them in the top 3, recall@3 = 0.5.

Precision @k: of the k chunks retrieved, how many were relevant? Top 3 with 1 relevant = 0.33. Low precision means noise gets fed to the LLM.

Mean Reciprocal Rank (MRR) : how high was the first correct chunk?
Score per question = 1 / (rank of first correct chunk), or 0 if not found.
Rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33.
Then we average all of them

Generation metrics: This needs LLM as judge
Faithfullness: Is every claim in the answer supported by the retrieved context? catches Hallucination

Ragas : It is an open-source library of ready-made RAG metrics, most of them LLM-judged.

Context Precision: Are the relevant chunks ranked near the top?
Context Recall : Did retrieval fetch everything needed to answer?
Response Relevancy: Does irrelevant context make the answer go wrong?
Faithfulness: Are the answer’s claims supported by the context?

====== E2 =======

E2 baseline (chunk 250/30, gemini-embedding-001, 573 chunks, 25 questions)
file level:  recall@3 0.98, MRR@10 0.98  (lenient: any chunk from the right file counts)
chunk level: recall@3 0.74, MRR@10 0.70  (strict: chunk must contain the evidence phrase)
Main miss pattern: right file, wrong chunk (q01, q05, q18, q20, q22, q24)

E5 ideas from the E2 misses:
- median chunk is 102 tokens (limit 250); 22% are under 50. Splitting on every
  heading plus removing code leaves one-sentence chunks.
- headings are removed from the chunk text, so chunks lose context.
- try: split only on # and ##, keep headings in the text (strip_headers=False),
  then rerun evaluate.py and compare against the baseline.

====== E3 ==========

| id  | judge | me | agree? | note |
|-----|-------|----|--------|------|
| q01 | 1     | 1  | yes    | HTTPException + raise in handling-errors-2 |


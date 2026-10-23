import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=6, initial_delay=2, max_delay=60),
    )
)
collection = chromadb.PersistentClient(path="chroma_db").get_collection("fastapi_docs")

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.1-flash-lite"
TOP_K = 3

SYSTEM_PROMPT = """You answer questions about FastAPI using ONLY the context provided.
If the context does not contain the answer, reply exactly: I don't know.
Do not use outside knowledge, even if you know the answer."""

def retrieve(question, k=TOP_K):
    vec = client.models.embed_content(
        model=EMBED_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    ).embeddings[0].values
    res = collection.query(query_embeddings=[vec], n_results=k)
    return [
        {"id": id_, "source": meta["source"], "text": doc}
        for id_, meta, doc in zip(res["ids"][0], res["metadatas"][0], res["documents"][0])
    ]

def answer(question):
    chunks = retrieve(question)
    context = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in chunks)
    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=f"Context:\n{context}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0)
    )
    return {"answer": response.text, "chunks": chunks}


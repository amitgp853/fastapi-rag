import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

q = "How do I return a 404 error?"
vec = client.models.embed_content(
    model="gemini-embedding-001", contents=q,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
).embeddings[0].values
res = chromadb.PersistentClient(path="chroma_db").get_collection("fastapi_docs").query(
    query_embeddings=[vec], n_results=3
)
for id_, meta in zip(res["ids"][0], res["metadatas"][0]):
    print(id_, meta["source"], meta.get("h2"))

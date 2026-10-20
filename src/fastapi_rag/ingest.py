import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

RAW_DIR = Path("data/raw")
CHUNK_SIZE = 250       # E5 will change these, so keep them as constants
CHUNK_OVERLAP = 30
COLLECTION = "fastapi_docs"

by_heading = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)
recursive = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

# drop the {* ../../docs_src/... *} code-include lines
def clean(text):
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("{*"))

ids, documents, metadatas = [], [], []

files = sorted(RAW_DIR.rglob("*.md"))
for path in files:
    source = str(path.relative_to(RAW_DIR))          # e.g. "security/oauth2-jwt.md"
    doc_id = source.removesuffix(".md").replace("/", "__")
    chunks = recursive.split_documents(by_heading.split_text(clean(path.read_text())))

    for i, chunk in enumerate(chunks):
        ids.append(f"{doc_id}-{i}")                  # stable id, E2 scores against this
        documents.append(chunk.page_content)
        meta = {"source": source, "chunk_id": i}
        meta.update(chunk.metadata)                  # h1 / h2 / h3
        metadatas.append(meta)

print(f"{len(files)} files -> {len(documents)} chunks")

load_dotenv()
client = genai.Client()

embeddings = []
for start in range(0, len(documents), 100):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=documents[start:start + 100],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    embeddings.extend(e.values for e in result.embeddings)
    print(f"Embedded {len(embeddings)} / {len(documents)}")
    if len(embeddings) < len(documents):
        time.sleep(60)

db = chromadb.PersistentClient(path="chroma_db")
try:
    db.delete_collection(COLLECTION)   # fresh index each run, no stale chunks after re-chunking
except Exception:
    pass
collection = db.create_collection(name=COLLECTION, configuration={"hnsw": {"space": "cosine"}})
collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
print("Stored in Chroma:", collection.count())

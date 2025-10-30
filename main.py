import os, json, argparse, pickle
from typing import Any
import numpy as np
from pydantic import BaseModel, Field
import faiss

from openai import OpenAI


class SourceItem(BaseModel):
    description: str = Field(
        ...,
        description=(
            "One-line, human-readable description of the source. This description should contain author of message"
        )
    )
    url: str = Field(..., description="HTTP/HTTPS link to the source message.")

class RAGAnswer(BaseModel):
    answer: str = Field(..., description="Concise answer synthesized from the retrieved messages only.")
    sources: list[SourceItem] = Field(
        ..., description="List of sources cited in the answer, ordered by relevance."
    )

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


def _get_index_path(artifacts_dir: str) -> str:
    return os.path.join(artifacts_dir, "index.flatip.bin")


def _get_meta_path(artifacts_dir: str) -> str:
    return os.path.join(artifacts_dir, "meta.pkl")


def embed_texts(texts: list[str], batch_size: int = 128) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    arr = np.array(out, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def save_artifacts(index: Any, meta: dict[str, Any], path: str = "artifacts_small") -> None:
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, _get_index_path(path))
    with open(_get_meta_path(path), "wb") as f:
        pickle.dump(meta, f)


def load_artifacts(path: str = "artifacts_small") -> tuple[Any, dict[str, Any]]:
    index = faiss.read_index(_get_index_path(path))
    with open(_get_meta_path(path), "rb") as f:
        meta = pickle.load(f)
    return index, meta


def build_index(messages_json: str, outdir: str = "artifacts_small") -> None:
    with open(messages_json, "r", encoding="utf-8") as f:
        msgs = json.load(f)

    texts = [m["content"] for m in msgs]
    urls = [m["url"] for m in msgs]
    ids = [m["message_id"] for m in msgs]
    authors = [m["author"] for m in msgs]

    embs = embed_texts(texts)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)

    meta = {"ids": ids, "urls": urls, "texts": texts, "authors": authors}
    save_artifacts(index, meta, outdir)
    print(f"OK: {index.ntotal} vectors, dim={d}")


def search(query: str, k: int = 5, path: str = "artifacts_small") -> list[dict[str, Any]]:
    index, meta = load_artifacts(path)
    q = embed_texts([query])
    D, I = index.search(q, k)
    out = []
    for rank, idx in enumerate(I[0]):
        out.append({
            "rank": rank + 1,
            "message_id": meta["ids"][idx],
            "url": meta["urls"][idx],
            "content": meta["texts"][idx],
            "author": meta["authors"][idx],
            "score": float(D[0][rank]),
        })
    return out


def format_rag_answer(answer: RAGAnswer) -> str:
    result = answer.answer
    result += "\n\nSources:\n\n"
    for source in answer.sources:
        result += f"- {source.description} ({source.url})\n\n"
    return result


def answer_with_sources(query: str, retrieved: list[dict[str, Any]]) -> RAGAnswer:
    context = ""
    for i, doc in enumerate(retrieved, 1):
        context += f"{i}. {doc['content']}\nAuthor: {doc['author']}\nSource: {doc['url']}\n\n"

    prompt = (
        "You are an assistant that MUST answer strictly based on the provided excerpts "
        "and MUST include source links.\n\n"
        f"Question: {query}\n\n"
        f"Excerpts:\n{context}\n"
        "Write a concise answer and provide sources with descriptions that include the author."
    )

    resp = client.beta.chat.completions.parse(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=RAGAnswer,
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.parsed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simple RAG CLI (FAISS + OpenAI)")
    ap.add_argument("--build", help="Path to messages.json to build the index")
    ap.add_argument("--ask", help="Ask a question against the built index")
    ap.add_argument("--artifacts", default="artifacts_small", help="Artifacts directory (index & meta)")
    args = ap.parse_args()

    if args.build:
        build_index(args.build, outdir=args.artifacts)
    if args.ask:
        hits = search(args.ask, k=5, path=args.artifacts)
        answer = answer_with_sources(args.ask, hits)
        formatted_answer = format_rag_answer(answer)
        print(formatted_answer)

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from openai import OpenAI

SUPPORTED_EXTENSIONS = {".md", ".txt"}
INDEX_VERSION = 2


@dataclass
class SearchHit:
    chunk_id: str
    source: str
    title: str
    text: str
    score: float
    semantic_score: float
    lexical_score: float
    source_boost: float


class RAGStore:
    def __init__(
        self,
        index_path: str = "data/kb_index.json",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.index_path = Path(index_path)
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.index: dict = self._empty_index()
        self.load()

    def _empty_index(self) -> dict:
        return {"version": INDEX_VERSION, "documents": {}, "chunks": []}

    def load(self) -> None:
        if self.index_path.exists():
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if "documents" not in data:
                data = self._upgrade_legacy_index(data)
            self.index = data
        else:
            self.index = self._empty_index()

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(self.index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self.index = self._empty_index()
        self.save()

    def get_chunk(self, chunk_id: str) -> dict | None:
        for chunk in self.index.get("chunks", []):
            if chunk["chunk_id"] == chunk_id:
                return chunk
        return None

    def list_sources(self) -> list[dict]:
        docs = []
        for source, meta in self.index.get("documents", {}).items():
            docs.append(
                {
                    "source": source,
                    "title": meta.get("title", source),
                    "chunks": meta.get("chunks", 0),
                    "updated_at": meta.get("updated_at"),
                    "hash": meta.get("hash"),
                }
            )
        docs.sort(key=lambda x: x["source"])
        return docs

    def ingest_directory(
        self,
        docs_dir: str,
        chunk_size: int = 700,
        chunk_overlap: int = 120,
    ) -> int:
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

        self.index = self._empty_index()
        for file_path in sorted(docs_path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self.add_or_update_document(
                    source=file_path.name,
                    content=file_path.read_text(encoding="utf-8"),
                    title=file_path.stem.replace("_", " ").strip().title(),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    persist=False,
                )
        self.save()
        return len(self.index["chunks"])

    def sync_directory(
        self,
        docs_dir: str,
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        delete_missing: bool = True,
    ) -> dict:
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

        seen_sources: set[str] = set()
        added = 0
        updated = 0
        skipped = 0

        for file_path in sorted(docs_path.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            content = file_path.read_text(encoding="utf-8")
            source = file_path.name
            seen_sources.add(source)
            content_hash = stable_hash(content)
            old = self.index.get("documents", {}).get(source)
            if old and old.get("hash") == content_hash:
                skipped += 1
                continue
            existed = source in self.index.get("documents", {})
            self.add_or_update_document(
                source=source,
                content=content,
                title=file_path.stem.replace("_", " ").strip().title(),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                persist=False,
            )
            if existed:
                updated += 1
            else:
                added += 1

        deleted = 0
        if delete_missing:
            existing_sources = set(self.index.get("documents", {}).keys())
            for source in sorted(existing_sources - seen_sources):
                self.delete_document(source, persist=False)
                deleted += 1

        self.save()
        return {
            "added": added,
            "updated": updated,
            "skipped": skipped,
            "deleted": deleted,
            "chunks": len(self.index.get("chunks", [])),
        }

    def add_or_update_document(
        self,
        source: str,
        content: str,
        title: str | None = None,
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        persist: bool = True,
    ) -> dict:
        normalized = normalize_text(content)
        title = title or Path(source).stem.replace("_", " ").strip().title()
        chunks = self.chunk_text(normalized, chunk_size=chunk_size, overlap=chunk_overlap)
        self.delete_document(source, persist=False)

        if not chunks:
            self.index.setdefault("documents", {})[source] = {
                "title": title,
                "hash": stable_hash(normalized),
                "chunks": 0,
                "updated_at": now_iso(),
            }
            if persist:
                self.save()
            return {"source": source, "chunks": 0}

        embeddings = self.embed_texts(chunks)
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings, strict=True), start=1):
            self.index.setdefault("chunks", []).append(
                {
                    "chunk_id": f"{Path(source).stem}-{idx}",
                    "source": source,
                    "title": title,
                    "text": chunk_text,
                    "embedding": embedding,
                    "token_hint": list(sorted(tokenize(chunk_text)))[:80],
                }
            )

        self.index.setdefault("documents", {})[source] = {
            "title": title,
            "hash": stable_hash(normalized),
            "chunks": len(chunks),
            "updated_at": now_iso(),
        }
        if persist:
            self.save()
        return {"source": source, "chunks": len(chunks)}

    def delete_document(self, source: str, persist: bool = True) -> bool:
        existed = source in self.index.get("documents", {})
        self.index["chunks"] = [c for c in self.index.get("chunks", []) if c.get("source") != source]
        self.index.get("documents", {}).pop(source, None)
        if persist:
            self.save()
        return existed

    def search(
        self,
        query: str,
        top_k: int = 4,
        source_filter: str | None = None,
    ) -> list[SearchHit]:
        if not self.index.get("chunks"):
            return []

        expanded_queries = expand_queries(query)
        query_embeddings = [np.array(self.embed_text(q), dtype=float) for q in expanded_queries]
        query_tokens = tokenize(" ".join(expanded_queries))
        wanted_source = source_filter.lower().strip() if source_filter else None

        scored_hits: list[SearchHit] = []
        for chunk in self.index["chunks"]:
            if wanted_source and wanted_source not in chunk["source"].lower():
                continue
            chunk_embedding = np.array(chunk["embedding"], dtype=float)
            semantic = max(cosine_similarity(qe, chunk_embedding) for qe in query_embeddings)
            lexical = lexical_overlap_score(query_tokens, set(chunk.get("token_hint") or tokenize(chunk["text"])))
            boost = source_match_boost(query, chunk["source"], chunk["title"])
            fused = semantic * 0.72 + lexical * 0.22 + boost * 0.06
            scored_hits.append(
                SearchHit(
                    chunk_id=chunk["chunk_id"],
                    source=chunk["source"],
                    title=chunk["title"],
                    text=chunk["text"],
                    score=float(fused),
                    semantic_score=float(semantic),
                    lexical_score=float(lexical),
                    source_boost=float(boost),
                )
            )

        scored_hits.sort(key=lambda x: (x.score, x.semantic_score, x.lexical_score), reverse=True)
        return scored_hits[:top_k]

    def search_by_source(self, source: str, keyword: str, top_k: int = 4) -> list[SearchHit]:
        return self.search(query=keyword, top_k=top_k, source_filter=source)

    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(input=text, model=self.embedding_model)
        return response.data[0].embedding

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for batch in batched(texts, batch_size):
            response = self.client.embeddings.create(input=batch, model=self.embedding_model)
            all_embeddings.extend([row.embedding for row in response.data])
        return all_embeddings

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
        text = normalize_text(text)
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 1 <= chunk_size:
                current = f"{current}\n{para}".strip()
                continue
            if current:
                chunks.append(current)
            if len(para) <= chunk_size:
                current = para
                continue
            # paragraph too long: fallback to sliding window on characters
            step = max(1, chunk_size - overlap)
            for start in range(0, len(para), step):
                piece = para[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
            current = ""
        if current:
            chunks.append(current)
        return dedupe_keep_order(chunks)

    def _upgrade_legacy_index(self, data: dict) -> dict:
        upgraded = self._empty_index()
        chunks = data.get("chunks", [])
        doc_count: dict[str, int] = {}
        for chunk in chunks:
            source = chunk.get("source", "unknown.txt")
            doc_count[source] = doc_count.get(source, 0) + 1
            chunk.setdefault("token_hint", list(sorted(tokenize(chunk.get("text", ""))))[:80])
            upgraded["chunks"].append(chunk)
        for source, count in doc_count.items():
            upgraded["documents"][source] = {
                "title": Path(source).stem.replace("_", " ").strip().title(),
                "hash": "legacy",
                "chunks": count,
                "updated_at": None,
            }
        return upgraded


def normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line.strip())


def batched(items: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if math.isclose(float(a_norm), 0.0) or math.isclose(float(b_norm), 0.0):
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def tokenize(text: str) -> set[str]:
    raw = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{1,8}", text.lower())
    tokens: set[str] = set(raw)
    for item in raw:
        if re.search(r"[\u4e00-\u9fff]", item) and len(item) > 1:
            tokens.update(item[i : i + 2] for i in range(len(item) - 1))
    return {tok for tok in tokens if tok.strip()}


def lexical_overlap_score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    overlap = query_tokens & doc_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(query_tokens)
    recall = len(overlap) / len(doc_tokens)
    return (2 * precision * recall) / (precision + recall + 1e-8)


def source_match_boost(query: str, source: str, title: str) -> float:
    q = query.lower()
    source_hit = 1.0 if source.lower() in q or Path(source).stem.lower() in q else 0.0
    title_hit = 1.0 if title.lower() in q else 0.0
    return max(source_hit, title_hit)


def expand_queries(query: str) -> list[str]:
    query = normalize_text(query)
    variants = {query}
    lowered = query.lower()
    rewrites = {
        "入职": ["入职流程", "onboarding"],
        "报销": ["费用报销", "expense reimbursement"],
        "值班": ["oncall", "故障值班"],
        "请假": ["休假", "leave policy"],
        "架构": ["系统设计", "architecture"],
        "规范": ["规则", "policy"],
    }
    for key, arr in rewrites.items():
        if key in lowered:
            variants.update(arr)
    return [q for q in variants if q]


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out

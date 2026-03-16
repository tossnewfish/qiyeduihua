from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from rag_store import RAGStore


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="构建或增量同步本地知识库索引")
    parser.add_argument("--docs", default="kb_docs", help="包含 .md/.txt 文件的目录")
    parser.add_argument("--index", default="data/kb_index.json", help="索引输出路径")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--sync", action="store_true", help="启用增量同步，而不是全量重建")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env before ingesting.")

    store = RAGStore(index_path=args.index, embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    if args.sync:
        stats = store.sync_directory(
            docs_dir=args.docs,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"Sync complete: {stats}")
    else:
        count = store.ingest_directory(
            docs_dir=args.docs,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"Indexed {count} chunks into {args.index}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from rag_store import RAGStore

REFUSAL_HINTS = ["没有足够", "不能确定", "不确定", "未找到", "依据不足"]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="评估知识库检索与问答效果")
    parser.add_argument("--cases", default="evals/sample_eval.json")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--with-generation", action="store_true", help="调用模型生成答案并检查拒答行为")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    store = RAGStore(index_path="data/kb_index.json", embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))

    retrieval_hits = 0
    answer_refusal_hits = 0
    rows = []

    for idx, case in enumerate(cases, start=1):
        hits = store.search(case["question"], top_k=args.top_k)
        retrieved_sources = [hit.source for hit in hits]
        expected_sources = set(case.get("expected_sources", []))
        retrieval_ok = not expected_sources or any(src in expected_sources for src in retrieved_sources)
        retrieval_hits += int(retrieval_ok)

        answer_text = ""
        refusal_ok = True
        if args.with_generation:
            from app import run_agent  # 延迟导入，避免无 API Key 时启动失败

            result = run_agent(session_id=f"eval-{idx}", user_message=case["question"], top_k=args.top_k)
            answer_text = result.answer
            if case.get("expect_refusal"):
                refusal_ok = any(hint in answer_text for hint in REFUSAL_HINTS)
            answer_refusal_hits += int(refusal_ok)

        rows.append(
            {
                "idx": idx,
                "question": case["question"],
                "retrieved_sources": retrieved_sources,
                "expected_sources": list(expected_sources),
                "retrieval_ok": retrieval_ok,
                "answer_preview": answer_text[:120],
                "refusal_ok": refusal_ok,
            }
        )

    summary = {
        "cases": len(cases),
        "retrieval_hit_rate": round(retrieval_hits / len(cases), 4) if cases else 0.0,
    }
    if args.with_generation:
        summary["refusal_pass_rate"] = round(answer_refusal_hits / len(cases), 4) if cases else 0.0

    print(json.dumps({"summary": summary, "details": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

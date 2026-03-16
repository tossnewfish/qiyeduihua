from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from prompts import FINAL_ANSWER_PROMPT, SUMMARY_PROMPT, SYSTEM_PROMPT
from rag_store import RAGStore

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", CHAT_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
SYSTEM_NAME = os.getenv("SYSTEM_NAME", "企业知识库问答 Agent")
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "8"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))
MAX_TOOL_STEPS = int(os.getenv("MAX_TOOL_STEPS", "4"))
KB_MIN_SCORE = float(os.getenv("KB_MIN_SCORE", "0.49"))
KB_WEAK_SCORE = float(os.getenv("KB_WEAK_SCORE", "0.58"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "45"))

app = FastAPI(title=SYSTEM_NAME, version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("agent_rag")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=REQUEST_TIMEOUT_SECONDS)
store = RAGStore(index_path="data/kb_index.json", embedding_model=EMBEDDING_MODEL)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=8)


class RebuildRequest(BaseModel):
    docs_dir: str = "kb_docs"
    chunk_size: int = 700
    chunk_overlap: int = 120


class SyncRequest(RebuildRequest):
    delete_missing: bool = True


class UpsertDocRequest(BaseModel):
    source: str = Field(..., min_length=1)
    title: str | None = None
    content: str = Field(..., min_length=1)
    chunk_size: int = 700
    chunk_overlap: int = 120


@dataclass
class AgentResult:
    session_id: str
    answer: str
    references: list[str]
    tool_traces: list[dict[str, Any]]
    retrieval_meta: dict[str, Any]
    trace_id: str


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def get(self, session_id: str) -> dict[str, Any]:
        return self._sessions.get(session_id, {"summary": "", "messages": []})

    def append(self, session_id: str, role: str, content: str) -> None:
        session = self._sessions.setdefault(session_id, {"summary": "", "messages": []})
        session["messages"].append({"role": role, "content": content})
        if len(session["messages"]) > MAX_HISTORY_TURNS * 2:
            old_messages = session["messages"][:-MAX_HISTORY_TURNS * 2]
            session["messages"] = session["messages"][-MAX_HISTORY_TURNS * 2 :]
            session["summary"] = summarize_messages(session.get("summary", ""), old_messages)

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


sessions = InMemorySessionStore()

TOOLS = [
    {
        "type": "function",
        "name": "search_kb",
        "description": "在企业知识库中做语义+关键词混合检索。适用于大多数知识问答。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用于检索的查询语句。"},
                "top_k": {"type": "integer", "description": "返回片段数量。"},
            },
            "required": ["query", "top_k"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_by_source",
        "description": "当用户指定了文档来源时，在指定来源里做更精确的检索。",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "目标文档文件名或其关键部分。"},
                "keyword": {"type": "string", "description": "来源内检索关键词。"},
                "top_k": {"type": "integer", "description": "返回片段数量。"},
            },
            "required": ["source", "keyword", "top_k"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "list_sources",
        "description": "列出当前知识库已收录的来源文档。适用于回答“有哪些制度/有哪些文档”。",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_chunk_by_id",
        "description": "根据 chunk_id 获取完整片段内容。",
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string", "description": "片段唯一编号。"}
            },
            "required": ["chunk_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


@app.get("/")
async def home() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "kb_chunks": len(store.index.get("chunks", [])),
        "kb_sources": len(store.index.get("documents", {})),
    }


@app.get("/admin/sources")
async def admin_sources() -> dict[str, Any]:
    return {"ok": True, "sources": store.list_sources()}


@app.post("/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    session_id = req.session_id or str(uuid.uuid4())
    result = run_agent(session_id=session_id, user_message=req.message, top_k=req.top_k)
    return {
        "session_id": result.session_id,
        "answer": result.answer,
        "references": result.references,
        "tool_traces": result.tool_traces,
        "retrieval_meta": result.retrieval_meta,
        "trace_id": result.trace_id,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    session_id = req.session_id or str(uuid.uuid4())
    trace_id = short_trace_id(session_id, req.message)

    def event_generator():
        result = run_agent(
            session_id=session_id,
            user_message=req.message,
            top_k=req.top_k,
            stream_final=True,
            forced_trace_id=trace_id,
        )
        yield sse("meta", {
            "session_id": session_id,
            "references": result.references,
            "trace_id": trace_id,
            "retrieval_meta": result.retrieval_meta,
        })
        for token in stream_answer_text(
            user_message=req.message,
            session_id=session_id,
            references=result.references,
            retrieval_meta=result.retrieval_meta,
            tool_traces=result.tool_traces,
            trace_id=trace_id,
        ):
            yield sse("delta", {"content": token})
        final_answer = collect_last_stream_answer(trace_id)
        persist_session(session_id, req.message, final_answer)
        yield sse("done", {
            "session_id": session_id,
            "references": result.references,
            "tool_traces": result.tool_traces,
            "retrieval_meta": result.retrieval_meta,
            "trace_id": trace_id,
        })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/admin/rebuild-index")
async def rebuild_index(req: RebuildRequest) -> dict[str, Any]:
    try:
        count = store.ingest_directory(
            docs_dir=req.docs_dir,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"ok": True, "chunks": count, "docs_dir": req.docs_dir}


@app.post("/admin/sync-index")
async def sync_index(req: SyncRequest) -> dict[str, Any]:
    try:
        stats = store.sync_directory(
            docs_dir=req.docs_dir,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            delete_missing=req.delete_missing,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, **stats, "docs_dir": req.docs_dir}


@app.post("/admin/upsert-document")
async def upsert_document(req: UpsertDocRequest) -> dict[str, Any]:
    result = store.add_or_update_document(
        source=req.source,
        title=req.title,
        content=req.content,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )
    return {"ok": True, **result}


@app.delete("/admin/document/{source}")
async def delete_document(source: str) -> dict[str, Any]:
    existed = store.delete_document(source)
    return {"ok": True, "deleted": existed, "source": source}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str) -> dict[str, Any]:
    sessions.clear(session_id)
    return {"ok": True, "session_id": session_id}


def run_agent(
    session_id: str,
    user_message: str,
    top_k: int,
    stream_final: bool = False,
    forced_trace_id: str | None = None,
) -> AgentResult:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="未配置 OPENAI_API_KEY。")

    trace_id = forced_trace_id or short_trace_id(session_id, user_message)
    if looks_like_prompt_injection(user_message):
        answer = "这条请求包含疑似提示词注入或越权指令，我不会执行其中的忽略规则、泄露系统信息或越权操作。你可以改为直接询问业务问题本身。"
        return AgentResult(session_id, answer, [], [], {"retrieval_status": "blocked"}, trace_id)

    history_bundle = build_history_bundle(session_id)
    input_messages = history_bundle["messages"] + [{"role": "user", "content": user_message}]
    tool_traces: list[dict[str, Any]] = []
    references: list[str] = []
    retrieval_meta: dict[str, Any] = {"retrieval_status": "not_used"}

    forced_search_done = False
    for step in range(MAX_TOOL_STEPS):
        response = safe_response_create(
            model=CHAT_MODEL,
            instructions=SYSTEM_PROMPT,
            input=input_messages,
            tools=TOOLS,
            parallel_tool_calls=False,
            tool_choice="auto",
            reasoning={"effort": "none"} if CHAT_MODEL.startswith("gpt-5") else None,
        )

        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            answer = response.output_text.strip()
            if not tool_traces and should_force_kb_search(user_message) and not forced_search_done:
                forced_search_done = True
                result = call_function("search_kb", {"query": user_message, "top_k": top_k}, top_k)
                retrieval_meta = result.get("retrieval_meta", retrieval_meta)
                references.extend(result.get("references", []))
                tool_traces.append({
                    "tool": "search_kb",
                    "args": {"query": user_message, "top_k": top_k},
                    "preview": preview_tool_result(result),
                    "fallback": True,
                })

            if stream_final:
                answer = ""
            elif tool_traces:
                answer = synthesize_answer(
                    user_message=user_message,
                    session_id=session_id,
                    references=sorted(set(references)),
                    retrieval_meta=retrieval_meta,
                    tool_traces=tool_traces,
                )
            if not stream_final:
                persist_session(session_id, user_message, answer)
            logger.info("trace=%s session=%s retrieval=%s refs=%s tools=%s", trace_id, session_id, retrieval_meta.get("retrieval_status"), sorted(set(references)), len(tool_traces))
            return AgentResult(session_id, answer, sorted(set(references)), tool_traces, retrieval_meta, trace_id)

        for output_item in response.output:
            dumped = output_item.model_dump(exclude_none=True)
            if dumped.get("type") in {"message", "reasoning", "function_call"}:
                input_messages.append(dumped)
            if dumped.get("type") != "function_call":
                continue

            name = dumped["name"]
            args = json.loads(dumped["arguments"])
            started = time.time()
            result = call_function(name=name, args=args, requested_top_k=top_k)
            elapsed_ms = round((time.time() - started) * 1000, 1)
            if name in {"search_kb", "search_by_source"}:
                references.extend(result.get("references", []))
                retrieval_meta = result.get("retrieval_meta", retrieval_meta)
            tool_traces.append(
                {
                    "tool": name,
                    "args": args,
                    "preview": preview_tool_result(result),
                    "elapsed_ms": elapsed_ms,
                }
            )
            input_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": dumped["call_id"],
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )

    raise HTTPException(status_code=500, detail="Agent reached max tool steps without producing a final answer.")


def call_function(name: str, args: dict[str, Any], requested_top_k: int) -> dict[str, Any]:
    if name == "search_kb":
        top_k = int(args.get("top_k") or requested_top_k)
        hits = store.search(query=args["query"], top_k=top_k)
        meta = build_retrieval_meta(hits)
        return {
            "count": len(hits),
            "references": [hit.source for hit in hits],
            "retrieval_meta": meta,
            "hits": [
                {
                    "chunk_id": hit.chunk_id,
                    "source": hit.source,
                    "title": hit.title,
                    "score": round(hit.score, 4),
                    "semantic_score": round(hit.semantic_score, 4),
                    "lexical_score": round(hit.lexical_score, 4),
                    "text": hit.text,
                }
                for hit in hits
            ],
        }

    if name == "search_by_source":
        top_k = int(args.get("top_k") or requested_top_k)
        hits = store.search_by_source(source=args["source"], keyword=args["keyword"], top_k=top_k)
        meta = build_retrieval_meta(hits)
        meta["source_filter"] = args["source"]
        return {
            "count": len(hits),
            "references": [hit.source for hit in hits],
            "retrieval_meta": meta,
            "hits": [
                {
                    "chunk_id": hit.chunk_id,
                    "source": hit.source,
                    "title": hit.title,
                    "score": round(hit.score, 4),
                    "semantic_score": round(hit.semantic_score, 4),
                    "lexical_score": round(hit.lexical_score, 4),
                    "text": hit.text,
                }
                for hit in hits
            ],
        }

    if name == "list_sources":
        sources = store.list_sources()
        return {"count": len(sources), "sources": sources}

    if name == "get_chunk_by_id":
        chunk = store.get_chunk(args["chunk_id"])
        return {"chunk": chunk, "found": chunk is not None}

    raise ValueError(f"Unsupported function: {name}")


def synthesize_answer(
    user_message: str,
    session_id: str,
    references: list[str],
    retrieval_meta: dict[str, Any],
    tool_traces: list[dict[str, Any]],
) -> str:
    if retrieval_meta.get("retrieval_status") == "blocked":
        return "当前请求触发了安全防护，我无法继续处理。"

    evidence_blocks: list[str] = []
    for trace in tool_traces:
        if trace["tool"] not in {"search_kb", "search_by_source", "get_chunk_by_id"}:
            continue
        evidence_blocks.append(trace["preview"])
    if not evidence_blocks and retrieval_meta.get("retrieval_status") in {"no_evidence", "weak_evidence"}:
        return "我没有在当前知识库中找到足够可靠的依据，因此不能确定回答。建议你补充更具体的文档名、制度名或关键词后再试。"

    history_bundle = build_history_bundle(session_id)
    synthesis_input = [
        {"role": "user", "content": build_synthesis_payload(user_message, history_bundle, references, retrieval_meta, evidence_blocks)},
    ]
    response = safe_response_create(
        model=CHAT_MODEL,
        instructions=FINAL_ANSWER_PROMPT,
        input=synthesis_input,
        reasoning={"effort": "none"} if CHAT_MODEL.startswith("gpt-5") else None,
    )
    answer = response.output_text.strip()
    if references and "参考来源" not in answer:
        answer += f"\n\n参考来源：{'、'.join(sorted(set(references)))}"
    return answer


def stream_answer_text(
    user_message: str,
    session_id: str,
    references: list[str],
    retrieval_meta: dict[str, Any],
    tool_traces: list[dict[str, Any]],
    trace_id: str,
):
    if retrieval_meta.get("retrieval_status") in {"blocked", "no_evidence"}:
        text = "我没有在当前知识库中找到足够可靠的依据，因此不能确定回答。建议补充更具体的文档名、制度名或关键词。"
        remember_stream_answer(trace_id, text)
        for chunk in chunk_text_stream(text):
            yield chunk
        return

    evidence_blocks: list[str] = []
    for trace in tool_traces:
        if trace["tool"] not in {"search_kb", "search_by_source", "get_chunk_by_id"}:
            continue
        evidence_blocks.append(trace["preview"])

    history_bundle = build_history_bundle(session_id)
    payload = build_synthesis_payload(user_message, history_bundle, references, retrieval_meta, evidence_blocks)
    full_text = ""
    with client.responses.stream(
        model=CHAT_MODEL,
        instructions=FINAL_ANSWER_PROMPT,
        input=[{"role": "user", "content": payload}],
        reasoning={"effort": "none"} if CHAT_MODEL.startswith("gpt-5") else None,
    ) as stream:
        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if not delta and hasattr(event, "model_dump"):
                    delta = event.model_dump().get("delta", "")
                if delta:
                    full_text += delta
                    yield delta
        final_response = stream.get_final_response()
        if not full_text:
            full_text = final_response.output_text.strip()
            for chunk in chunk_text_stream(full_text):
                yield chunk
    if references and "参考来源" not in full_text:
        suffix = f"\n\n参考来源：{'、'.join(sorted(set(references)))}"
        full_text += suffix
        yield suffix
    remember_stream_answer(trace_id, full_text)


def persist_session(session_id: str, user_message: str, answer: str) -> None:
    sessions.append(session_id, "user", user_message)
    sessions.append(session_id, "assistant", answer)


def safe_response_create(**kwargs: Any):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    last_error = None
    for attempt in range(2):
        try:
            return client.responses.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("openai call failed attempt=%s error=%s", attempt + 1, exc)
            time.sleep(0.8 * (attempt + 1))
    raise HTTPException(status_code=502, detail=f"模型调用失败：{last_error}") from last_error


def summarize_messages(existing_summary: str, messages: list[dict[str, str]]) -> str:
    if not messages:
        return existing_summary
    dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    prompt = f"已有摘要：\n{existing_summary or '无'}\n\n新增对话：\n{dialogue}"
    try:
        response = safe_response_create(
            model=SUMMARY_MODEL,
            instructions=SUMMARY_PROMPT,
            input=[{"role": "user", "content": prompt}],
        )
        return response.output_text.strip()
    except Exception:  # noqa: BLE001
        return existing_summary


def build_history_bundle(session_id: str) -> dict[str, Any]:
    session = sessions.get(session_id)
    items = []
    if session.get("summary"):
        items.append({"role": "system", "content": f"历史摘要：{session['summary']}"})
    items.extend(session.get("messages", []))
    return {"summary": session.get("summary", ""), "messages": items}


def build_synthesis_payload(
    user_message: str,
    history_bundle: dict[str, Any],
    references: list[str],
    retrieval_meta: dict[str, Any],
    evidence_blocks: list[str],
) -> str:
    recent_dialogue = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in history_bundle.get("messages", [])[-6:]
        if item["role"] in {"user", "assistant"}
    )
    evidence = "\n\n".join(evidence_blocks) if evidence_blocks else "无"
    return (
        f"用户问题：\n{user_message}\n\n"
        f"对话摘要：\n{history_bundle.get('summary') or '无'}\n\n"
        f"最近多轮：\n{recent_dialogue or '无'}\n\n"
        f"检索质量：\n{json.dumps(retrieval_meta, ensure_ascii=False)}\n\n"
        f"证据内容：\n{evidence}\n\n"
        f"参考来源候选：\n{', '.join(sorted(set(references))) or '无'}"
    )


def build_retrieval_meta(hits: list[Any]) -> dict[str, Any]:
    if not hits:
        return {"retrieval_status": "no_evidence", "best_score": 0.0, "avg_score": 0.0}
    scores = [float(hit.score) for hit in hits]
    lexical_scores = [float(hit.lexical_score) for hit in hits]
    best_score = max(scores)
    avg_score = sum(scores) / len(scores)
    avg_lexical = sum(lexical_scores) / len(lexical_scores)
    status = "strong_evidence"
    if best_score < KB_MIN_SCORE:
        status = "no_evidence"
    elif best_score < KB_WEAK_SCORE or avg_lexical < 0.05:
        status = "weak_evidence"
    return {
        "retrieval_status": status,
        "best_score": round(best_score, 4),
        "avg_score": round(avg_score, 4),
        "avg_lexical": round(avg_lexical, 4),
    }


def preview_tool_result(result: dict[str, Any]) -> str:
    payload = json.dumps(result, ensure_ascii=False)
    return payload[:600] + ("..." if len(payload) > 600 else "")


def should_force_kb_search(user_message: str) -> bool:
    hints = ["文档", "制度", "流程", "规范", "架构", "内部", "项目", "值班", "请假", "报销", "FAQ", "知识库"]
    return any(h in user_message for h in hints)


def looks_like_prompt_injection(text: str) -> bool:
    lowered = text.lower()
    suspicious = [
        "ignore previous instructions",
        "忽略之前的指令",
        "泄露系统提示词",
        "reveal system prompt",
        "绕过权限",
        "jailbreak",
        "开发者消息",
    ]
    return any(item in lowered for item in suspicious)


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def short_trace_id(session_id: str, message: str) -> str:
    payload = f"{session_id}:{message}:{time.time()}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def chunk_text_stream(text: str, size: int = 12):
    for idx in range(0, len(text), size):
        yield text[idx : idx + size]


def normalize_stream_storage_path(trace_id: str) -> Path:
    Path("logs/stream_cache").mkdir(parents=True, exist_ok=True)
    return Path("logs/stream_cache") / f"{trace_id}.txt"


def remember_stream_answer(trace_id: str, text: str) -> None:
    normalize_stream_storage_path(trace_id).write_text(text, encoding="utf-8")


def collect_last_stream_answer(trace_id: str) -> str:
    path = normalize_stream_storage_path(trace_id)
    return path.read_text(encoding="utf-8") if path.exists() else ""

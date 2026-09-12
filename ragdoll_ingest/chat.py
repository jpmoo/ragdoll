"""Insight chat: talk with the collection to correct and steer it.

The chat is how insights get fixed. It reads the same record the nightly run wrote — insights, their lineage,
the reflections, the passages behind them — and it can create, edit, retire, restore and undo, plus set
standing guidelines that change what future runs generate at all.

It acts when told to, rather than proposing changes for approval: everything it does is recorded as a revision
with the reason, retiring keeps the insight, and the last edit can be undone. Changes made here are made as the
actor "chat", which pins the insight so automated runs leave it alone.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from . import config
from .insights import (
    INSIGHTS_GROUP,
    InsightError,
    _connect_insights,
    add_guideline,
    create_insight,
    find_similar_insights,
    get_insight,
    list_guidelines,
    list_insights,
    remove_guideline,
    restore_insight,
    retire_insight,
    revert_last_edit,
    update_insight,
)
from .embedder import embed
from .stew import get_run, list_runs, read_reflection

logger = logging.getLogger(__name__)

ACTOR = "chat"
# Tools that change something; used to tell the user what the turn actually did
WRITING_TOOLS = (
    "create_insight", "edit_insight", "retire_insight", "restore_insight", "undo_last_edit",
    "add_guideline", "remove_guideline",
)

SYSTEM_PROMPT = """You are RAGDoll's insight chat. You are talking with the person who owns these document \
collections, about the insights RAGDoll has built from how they get used.

An insight is a durable claim with a statement, the question it answers, a rationale, and lineage: the passages, \
queries and other insights behind it. The nightly run creates them; you are how they get corrected.

What the fields mean:
- origin "stew": the nightly run wrote it, from questions people asked and the passages those questions returned.
- origin "chat": you wrote it here, grounded in passages you found.
- origin "asserted": the owner stated it directly; it needs no passage support.
- origin "agent": an assistant submitted it over MCP.
- pinned: a person has changed it, so automated runs leave it alone.
- status "retired" or "superseded": no longer searched, but kept with the reason so runs don't regenerate it.
- lineage: the evidence actually recorded. An insight with no chunk lineage is not grounded in the documents, \
whatever its rationale claims.

How to work:
- Look things up before you talk about them. Use search_insights and get_insight for what is stored, \
search_sources for what the documents actually say, and read_reflection for why a run concluded something.
- Act when the owner clearly asks. Edit, retire, restore, undo, or add a guideline, then say plainly what you \
changed and its id. Ask first only when the instruction is ambiguous, not to be cautious.
- Nothing here is destructive: retiring keeps the insight and its reason, and undo_last_edit reverses an edit. \
Say so when it is relevant, briefly.
- When the owner tells you something is wrong, ask yourself whether it is one insight or a pattern. A pattern \
belongs in a guideline, which steers what future runs generate at all.
- Insights you create from what the owner asserts are marked "asserted"; insights you create from passages you \
found are "chat" and should cite them in the rationale.
- Never invent a statement, a source or an id. If the record does not support something, say so.
- Be concise and direct. Say what you did or found, not what you are about to do."""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_insights",
            "description": "Find stored insights by meaning. Use before discussing or changing any insight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for"},
                    "status": {"type": "string", "enum": ["active", "retired", "superseded", "all"]},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_insight",
            "description": "One insight in full: statement, rationale, lineage (passages, queries) and revision history.",
            "parameters": {
                "type": "object",
                "properties": {"insight_id": {"type": "integer"}},
                "required": ["insight_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_sources",
            "description": "Search the document collections (not insights) to check what the sources actually say.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "collections": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_runs",
            "description": "Recent nightly runs with their counts.",
            "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_reflection",
            "description": "A run's written account of what it looked at, concluded and set aside.",
            "parameters": {
                "type": "object",
                "properties": {"run_id": {"type": "string"}},
                "required": ["run_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_insight",
            "description": "Record a new insight. Use origin 'asserted' for something the owner states, 'chat' when you ground it in passages you found.",
            "parameters": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "description": "The learning, stated plainly and standing on its own"},
                    "question": {"type": "string", "description": "The question it answers"},
                    "rationale": {"type": "string", "description": "Why it holds; name the documents if you found them"},
                    "topic": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "origin": {"type": "string", "enum": ["asserted", "chat"]},
                },
                "required": ["statement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_insight",
            "description": "Change an insight's wording or metadata. Only pass the fields that change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "insight_id": {"type": "integer"},
                    "statement": {"type": "string"},
                    "question": {"type": "string"},
                    "rationale": {"type": "string"},
                    "topic": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string", "description": "Why, in the owner's terms"},
                },
                "required": ["insight_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retire_insight",
            "description": "Stop an insight being searched. It is kept, with the reason, and future runs will not regenerate it.",
            "parameters": {
                "type": "object",
                "properties": {"insight_id": {"type": "integer"}, "reason": {"type": "string"}},
                "required": ["insight_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restore_insight",
            "description": "Make a retired insight searchable again.",
            "parameters": {
                "type": "object",
                "properties": {"insight_id": {"type": "integer"}, "reason": {"type": "string"}},
                "required": ["insight_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo_last_edit",
            "description": "Put an insight's text back the way it was before the last edit.",
            "parameters": {
                "type": "object",
                "properties": {"insight_id": {"type": "integer"}, "reason": {"type": "string"}},
                "required": ["insight_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_guidelines",
            "description": "Standing instructions the nightly run follows.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_guideline",
            "description": "Add a standing instruction for future nightly runs, e.g. what not to record or how to phrase insights.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_guideline",
            "description": "Deactivate a standing instruction.",
            "parameters": {
                "type": "object",
                "properties": {"guideline_id": {"type": "integer"}},
                "required": ["guideline_id"],
            },
        },
    },
]


def init_chat_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            last_active TEXT NOT NULL,
            title TEXT
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES chat_sessions(id),
            ts TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_name TEXT,
            tool_args TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_chat_messages_session ON chat_messages(session_id);
    """)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _conn() -> sqlite3.Connection:
    conn = _connect_insights()
    init_chat_db(conn)
    return conn


def new_session(title: str | None = None) -> int:
    conn = _conn()
    try:
        now = _now()
        cur = conn.execute(
            "INSERT INTO chat_sessions (started_at, last_active, title) VALUES (?, ?, ?)", (now, now, title)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM chat_messages m WHERE m.session_id = s.id AND m.role = 'user') "
            "AS messages FROM chat_sessions s ORDER BY s.last_active DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_messages(session_id: int, limit: int | None = None) -> list[dict[str, Any]]:
    """Stored turns for a session, oldest first. Tool calls are kept so the record shows what was done."""
    conn = _conn()
    try:
        sql = "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id"
        rows = conn.execute(sql, (session_id,)).fetchall()
        out = [dict(r) for r in rows]
        return out[-limit:] if limit else out
    finally:
        conn.close()


def _save_message(session_id: int, role: str, content: str | None, tool_name: str | None = None,
                  tool_args: Any = None) -> None:
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO chat_messages (session_id, ts, role, content, tool_name, tool_args) VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id, _now(), role, content, tool_name,
                json.dumps(tool_args, ensure_ascii=False) if tool_args is not None else None,
            ),
        )
        conn.execute("UPDATE chat_sessions SET last_active = ? WHERE id = ?", (_now(), session_id))
        conn.commit()
    finally:
        conn.close()


# --- tools ---------------------------------------------------------------

def _brief(insight: dict[str, Any]) -> dict[str, Any]:
    keep = ("id", "statement", "question", "topic", "tags", "origin", "status", "status_reason", "confidence",
            "pinned", "created_at", "updated_at", "similarity")
    return {k: insight[k] for k in keep if k in insight}


def _tool_search_insights(query: str, status: str = "active", limit: int = 5) -> dict[str, Any]:
    statuses = ("active", "retired", "superseded") if status == "all" else (status,)
    emb = embed([query], group=INSIGHTS_GROUP)[0]
    found = find_similar_insights(emb, statuses=statuses, top_k=min(limit, 10), min_similarity=0.3)
    return {"insights": [_brief(i) for i in found]}


def _tool_get_insight(insight_id: int) -> dict[str, Any]:
    i = get_insight(insight_id)
    lineage = [
        {
            "kind": e["kind"], "relation": e["relation"], "source": e["ref_source_path"], "group": e["ref_group"],
            "query_id": e["ref_query_id"], "insight_id": e["ref_insight_id"],
            "text": (e["text_snapshot"] or "")[:400],
        }
        for e in i["lineage"]
    ]
    revisions = [
        {"ts": r["ts"], "actor": r["actor"], "action": r["action"], "reason": r["reason"]} for r in i["revisions"]
    ]
    return {**_brief(i), "rationale": i["rationale"], "open_questions": i["open_questions"],
            "run_id": i["run_id"], "version": i["version"], "lineage": lineage, "revisions": revisions}


def _tool_search_sources(query: str, collections: list[str] | None = None, limit: int = 8) -> dict[str, Any]:
    # Imported here, not at module scope: the API module pulls in FastAPI, which the terminal chat never needs
    from .api import _do_query

    # log_as=None: the chat's own lookups are not the usage the nightly run learns from
    result = _do_query(query, None, config.QUERY_THRESHOLD, collections or None, include_insights=False, log_as=None)
    hits = result.get("results") or []
    return {
        "passages": [
            {"group": h["group"], "source": h["source_name"], "chunk_id": h["chunk_id"],
             "similarity": h["similarity"], "text": (h["text"] or "")[:800]}
            for h in hits[:limit]
        ],
        "total_matching": len(hits),
    }


def _tool_list_runs(limit: int = 5) -> dict[str, Any]:
    return {"runs": [
        {k: r[k] for k in ("run_id", "started_at", "model", "dry_run", "n_queries", "n_created", "n_reinforced",
                           "n_rejected")}
        for r in list_runs(limit)
    ]}


def _tool_read_reflection(run_id: str) -> dict[str, Any]:
    text = read_reflection(run_id)
    if text is None:
        run = get_run(run_id)
        return {"error": f"No reflection for run {run_id}" + ("" if run else " (no such run)")}
    return {"run_id": run_id, "reflection": text[:8000]}


def _tool_create_insight(statement: str, question: str | None = None, rationale: str | None = None,
                         topic: str | None = None, tags: list[str] | None = None, confidence: float | None = None,
                         origin: str = "asserted", session_id: int | None = None) -> dict[str, Any]:
    if origin not in ("asserted", "chat"):
        origin = "asserted"
    created = create_insight(
        statement, origin=origin, actor=ACTOR, question=question, rationale=rationale, topic=topic, tags=tags,
        confidence=confidence, reason="Created in chat", session_id=str(session_id) if session_id else None,
    )
    return {"created": _brief(created)}


def _tool_edit_insight(insight_id: int, reason: str, session_id: int | None = None, **fields: Any) -> dict[str, Any]:
    changes = {k: v for k, v in fields.items() if v is not None}
    if not changes:
        return {"error": "No fields to change"}
    updated = update_insight(
        insight_id, changes, actor=ACTOR, reason=reason, session_id=str(session_id) if session_id else None
    )
    return {"updated": _brief(updated), "changed_fields": sorted(changes)}


def _tool_retire_insight(insight_id: int, reason: str, session_id: int | None = None) -> dict[str, Any]:
    return {"retired": _brief(retire_insight(
        insight_id, actor=ACTOR, reason=reason, session_id=str(session_id) if session_id else None))}


def _tool_restore_insight(insight_id: int, reason: str | None = None, session_id: int | None = None) -> dict[str, Any]:
    return {"restored": _brief(restore_insight(
        insight_id, actor=ACTOR, reason=reason, session_id=str(session_id) if session_id else None))}


def _tool_undo_last_edit(insight_id: int, reason: str | None = None, session_id: int | None = None) -> dict[str, Any]:
    return {"reverted": _brief(revert_last_edit(
        insight_id, actor=ACTOR, reason=reason, session_id=str(session_id) if session_id else None))}


def _tool_list_guidelines() -> dict[str, Any]:
    return {"guidelines": [{"id": g["id"], "text": g["text"]} for g in list_guidelines()]}


def _tool_add_guideline(text: str) -> dict[str, Any]:
    return {"added": add_guideline(text, actor=ACTOR)}


def _tool_remove_guideline(guideline_id: int) -> dict[str, Any]:
    g = remove_guideline(guideline_id)
    return {"removed": {"id": g["id"], "text": g["text"]}}


# Tools that need to know which chat session they were called from
_SESSION_TOOLS = {"create_insight", "edit_insight", "retire_insight", "restore_insight", "undo_last_edit"}

_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "search_insights": _tool_search_insights,
    "get_insight": _tool_get_insight,
    "search_sources": _tool_search_sources,
    "list_runs": _tool_list_runs,
    "read_reflection": _tool_read_reflection,
    "create_insight": _tool_create_insight,
    "edit_insight": _tool_edit_insight,
    "retire_insight": _tool_retire_insight,
    "restore_insight": _tool_restore_insight,
    "undo_last_edit": _tool_undo_last_edit,
    "list_guidelines": _tool_list_guidelines,
    "add_guideline": _tool_add_guideline,
    "remove_guideline": _tool_remove_guideline,
}


def _run_tool(name: str, args: dict[str, Any], session_id: int) -> dict[str, Any]:
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"No such tool: {name}"}
    kwargs = dict(args or {})
    if name in _SESSION_TOOLS:
        kwargs["session_id"] = session_id
    try:
        return fn(**kwargs)
    except InsightError as e:
        # Expected refusals (missing id, wrong status) are answers, not failures: let the model explain them
        return {"error": str(e)}
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}
    except Exception as e:
        logger.exception("Chat tool %s failed", name)
        return {"error": f"{name} failed: {e}"}


# --- the loop ------------------------------------------------------------

def opening_context() -> str:
    """What the owner should know before saying anything: runs since we last talked, and what stands now."""
    runs = list_runs(3)
    insights = list_insights(limit=5)
    guidelines = list_guidelines()
    lines = []
    if runs:
        lines.append("Recent runs:")
        for r in runs:
            mode = "dry run" if r["dry_run"] else "wrote"
            lines.append(
                f"- {r['run_id']} ({mode}, {r['model']}): {r['n_queries']} queries, {r['n_created']} created, "
                f"{r['n_reinforced']} reinforced, {r['n_rejected']} rejected"
            )
    if insights:
        lines.append("\nMost recently updated insights:")
        for i in insights:
            lines.append(f"- [{i['id']}] ({i['origin']}) {i['statement'][:140]}")
    if guidelines:
        lines.append("\nStanding instructions:")
        lines.extend(f"- [{g['id']}] {g['text']}" for g in guidelines)
    return "\n".join(lines) or "Nothing recorded yet."


def _chat_call(messages: list[dict[str, Any]], model: str) -> dict[str, Any] | None:
    url = (config.CHAT_OLLAMA_HOST or "").rstrip("/")
    try:
        r = requests.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "tools": TOOLS,
                "stream": False,
                # Thinking models otherwise answer in a field the caller never reads
                "think": False,
                "options": {"temperature": 0.3, "num_ctx": config.CHAT_NUM_CTX},
            },
            timeout=config.CHAT_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("message") or {}
    except Exception as e:
        logger.warning("Chat model call failed (model=%s): %s", model, e)
        return None


def _history_for_model(session_id: int) -> list[dict[str, Any]]:
    """Replay recent turns. Tool traffic is summarized rather than replayed in full to keep the window usable."""
    messages: list[dict[str, Any]] = []
    for m in get_messages(session_id, limit=config.CHAT_HISTORY_MESSAGES):
        if m["role"] in ("user", "assistant") and m["content"]:
            messages.append({"role": m["role"], "content": m["content"]})
        elif m["role"] == "tool":
            messages.append({
                "role": "assistant",
                "content": f"(used {m['tool_name']}: {(m['content'] or '')[:400]})",
            })
    return messages


def ask(
    session_id: int,
    text: str,
    *,
    model: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """One turn: the model may call tools, then answers. Returns its reply and what it changed."""
    model = model or config.CHAT_MODEL
    say = progress or (lambda _m: None)
    _save_message(session_id, "user", text)

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": "Current state:\n" + opening_context()}]
    messages += _history_for_model(session_id)

    actions: list[dict[str, Any]] = []
    for _step in range(config.CHAT_MAX_STEPS):
        msg = _chat_call(messages, model)
        if msg is None:
            reply = "I couldn't reach the chat model. Nothing was changed."
            _save_message(session_id, "assistant", reply)
            return {"reply": reply, "actions": actions, "error": "model_unreachable"}

        calls = msg.get("tool_calls") or []
        if not calls:
            reply = (msg.get("content") or "").strip() or (msg.get("thinking") or "").strip()
            reply = reply or "(no reply)"
            _save_message(session_id, "assistant", reply)
            return {"reply": reply, "actions": actions}

        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            say(f"{name}({', '.join(f'{k}={v!r}' for k, v in list(args.items())[:3])})")
            result = _run_tool(name, args, session_id)
            if name in WRITING_TOOLS and "error" not in result:
                actions.append({"tool": name, "args": args, "result": result})
            _save_message(session_id, "tool", json.dumps(result, ensure_ascii=False)[:4000], name, args)
            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, ensure_ascii=False)[:6000],
            })

    reply = "I stopped after too many steps without answering. Nothing further was changed."
    _save_message(session_id, "assistant", reply)
    return {"reply": reply, "actions": actions, "error": "max_steps"}

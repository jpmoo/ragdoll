"""Nightly "stew": read the query log, cluster what people asked, and synthesize insights from the passages
those questions returned.

The run is a dry run by default: it records what it *would* create (candidates, the passages they cite, the
alternatives it rejected) and writes a reflection, but creates no insights. Pass write=True to create them.

Everything is kept so the reflection and the later chat can explain how an insight came about:
- stew_runs / stew_clusters / stew_candidates tables in the insights DB
- a markdown reflection under {DATA_DIR}/insights/reflections/<run_id>.md
"""

import json
import logging
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from . import config
from .action_log import log as action_log
from .embedder import cosine_similarity, embed
from .insights import (
    INSIGHTS_GROUP,
    _connect_insights,
    create_insight,
    find_similar_insights,
    reinforce_insight,
)
from .query_log import hits_for_queries, queries_since

logger = logging.getLogger(__name__)

# Passage text given to the model per chunk (snapshots in the query log can be longer)
EVIDENCE_MAX_CHARS = 1200
# Existing insights shown to the model as "already recorded" / "already rejected"
MAX_EXISTING_INSIGHTS = 8
MAX_REJECTED_INSIGHTS = 5
# Candidates accepted from one cluster
MAX_CANDIDATES_PER_CLUSTER = 3


def init_stew_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stew_runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            model TEXT,
            dry_run INTEGER NOT NULL,
            since TEXT,
            n_queries INTEGER DEFAULT 0,
            n_clusters INTEGER DEFAULT 0,
            n_candidates INTEGER DEFAULT 0,
            n_created INTEGER DEFAULT 0,
            n_reinforced INTEGER DEFAULT 0,
            n_rejected INTEGER DEFAULT 0,
            reflection_path TEXT,
            status TEXT,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS stew_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            label TEXT,
            query_ids TEXT,
            n_queries INTEGER,
            n_chunks INTEGER,
            collections TEXT,
            outcome TEXT,
            notes TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_stew_clusters_run ON stew_clusters(run_id);

        CREATE TABLE IF NOT EXISTS stew_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            cluster_id INTEGER,
            statement TEXT NOT NULL,
            question TEXT,
            rationale TEXT,
            topic TEXT,
            tags TEXT,
            confidence REAL,
            trace TEXT,
            decision TEXT,
            decision_reason TEXT,
            insight_id INTEGER,
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_stew_candidates_run ON stew_candidates(run_id);
    """)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _new_run_id() -> str:
    """Sortable id with a short random suffix, so two runs in the same second don't collide."""
    return datetime.now(timezone.utc).strftime("stew-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:4]


def _reflections_dir() -> Path:
    d = config.get_group_paths(INSIGHTS_GROUP).group_dir / "reflections"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _default_since(conn: sqlite3.Connection) -> str:
    """Start from the last finished run, or STEW_LOOKBACK_DAYS ago if this is the first one."""
    row = conn.execute(
        "SELECT started_at FROM stew_runs WHERE status = 'ok' ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if row and row["started_at"]:
        return row["started_at"]
    return (datetime.now(timezone.utc) - timedelta(days=config.STEW_LOOKBACK_DAYS)).strftime("%Y-%m-%d %H:%M:%S")


def _cluster_queries(queries: list[dict[str, Any]], threshold: float) -> list[list[dict[str, Any]]]:
    """Group queries by embedding similarity: a query joins the cluster holding the most similar question.

    Comparing against the closest member rather than a cluster average keeps a question that is phrased
    differently from the rest of the group but clearly about the same thing. Volumes here are small (tens to
    hundreds of queries a night), so the simple O(n^2) pass is fine.
    """
    clusters: list[list[dict[str, Any]]] = []
    for q in queries:
        emb = q.get("embedding")
        if not emb:
            continue
        best, best_sim = None, threshold
        for c in clusters:
            sim = max(cosine_similarity(emb, m["embedding"]) for m in c)
            if sim >= best_sim:
                best, best_sim = c, sim
        if best is None:
            clusters.append([q])
        else:
            best.append(q)
    return clusters


def _centroid(vectors: list[list[float]]) -> list[float]:
    return [sum(col) / len(vectors) for col in zip(*vectors)]


def _cluster_label(members: list[dict[str, Any]]) -> str:
    """Shortest prompt in the cluster reads best as its name."""
    prompt = min((m["prompt"] for m in members), key=len)
    return prompt if len(prompt) <= 120 else prompt[:117] + "..."


def _gather_evidence(members: list[dict[str, Any]], max_chunks: int) -> tuple[list[dict[str, Any]], list[int]]:
    """Deduplicated document passages these queries returned (best similarity wins), plus insight ids seen.

    Passages come from the query log's snapshots rather than the collections, so the evidence is what the
    asker actually saw at the time.
    """
    best: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    insight_ids: list[int] = []
    for q in members:
        for h in q.get("hits") or []:
            if h["group_name"] == INSIGHTS_GROUP:
                m = re.match(r"insight/(\d+)$", h["source_path"] or "")
                if m and int(m.group(1)) not in insight_ids:
                    insight_ids.append(int(m.group(1)))
                continue
            key = (h["group_name"], h["source_path"], h["chunk_index"])
            prev = best.get(key)
            if prev is None or h["similarity"] > prev["similarity"]:
                best[key] = {
                    "group": h["group_name"],
                    "source_path": h["source_path"],
                    "source_name": Path(h["source_path"]).name,
                    "chunk_id": h["chunk_id"],
                    "chunk_index": h["chunk_index"],
                    "chunk_role": h["chunk_role"],
                    "similarity": h["similarity"],
                    "text": (h["text_snapshot"] or "")[:EVIDENCE_MAX_CHARS],
                    "query_id": q["id"],
                }
    evidence = sorted(best.values(), key=lambda e: e["similarity"], reverse=True)[:max_chunks]
    for i, e in enumerate(evidence, 1):
        e["n"] = i
    return evidence, insight_ids


def _format_prompt(
    label: str,
    members: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> str:
    questions = "\n".join(f"- {m['prompt']} ({m['ts']} UTC)" for m in members)
    passages = "\n\n".join(
        f"[{e['n']}] {e['group']} / {e['source_name']} (similarity {e['similarity']:.2f})\n{e['text']}"
        for e in evidence
    )
    existing_block = "\n".join(f"- I{i['id']}: {i['statement']}" for i in existing) or "- (none)"
    rejected_block = (
        "\n".join(f"- {i['statement']} — rejected because: {i['status_reason'] or 'no reason recorded'}" for i in rejected)
        or "- (none)"
    )
    return (
        "You are RAGDoll's nightly synthesis pass. RAGDoll is a retrieval system over a person's document "
        "collections. Below are questions that were asked recently, the passages the search returned for them, "
        "insights already recorded, and statements previously rejected.\n\n"
        "Your job is to propose insights worth remembering: durable claims, grounded in the passages, that answer "
        "the kind of question being asked. Quality matters far more than quantity.\n\n"
        "Rules:\n"
        f"- Cite the passages that support each insight by their [number]. At least {config.STEW_MIN_SUPPORT} "
        "distinct passages are required. Never cite a number that is not listed below.\n"
        "- Do not restate an existing insight. If your point only reinforces one, put its id in builds_on and make "
        "the statement about what is new.\n"
        "- Do not restate a rejected statement.\n"
        "- Prefer claims that connect passages from different documents or collections.\n"
        "- Say what would make you doubt the claim in weak_spots, and what other readings you set aside in "
        "alternatives_considered.\n"
        f"- Return at most {MAX_CANDIDATES_PER_CLUSTER} insights. If nothing here is worth recording, return an "
        "empty list and explain why in notes.\n\n"
        f"QUESTIONS ASKED (theme: {label})\n{questions}\n\n"
        f"PASSAGES RETURNED\n{passages}\n\n"
        f"INSIGHTS ALREADY RECORDED\n{existing_block}\n\n"
        f"PREVIOUSLY REJECTED\n{rejected_block}\n\n"
        "Reply with JSON only, in this shape:\n"
        '{"insights": [{"statement": "one or two sentences, stated plainly", '
        '"question": "the question this answers", "rationale": "why it holds, referring to the passages", '
        '"topic": "short title", "tags": ["..."], "confidence": 0.0-1.0, "supports": [1, 4], '
        '"builds_on": [insight ids], "tensions_with": [insight ids], '
        '"alternatives_considered": "other readings you set aside", "weak_spots": "what is thin here"}], '
        '"notes": "what you considered and left out"}'
    )


def _generate(prompt: str, model: str, *, want_json: bool, temperature: float = 0.2) -> str | None:
    """One call to the insight model. Returns the response text, or None if the call failed.

    Every model call in the stew goes through here so the host, context size and timeout are set in one place.
    """
    url = (config.INSIGHT_OLLAMA_HOST or "").rstrip("/")
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Ollama's default context is far smaller than these prompts and would truncate them silently
        "options": {"temperature": temperature, "num_ctx": config.INSIGHT_NUM_CTX},
    }
    if want_json:
        payload["format"] = "json"
    try:
        r = requests.post(f"{url}/api/generate", json=payload, timeout=config.INSIGHT_TIMEOUT)
        r.raise_for_status()
        return (r.json().get("response") or "").strip() or None
    except Exception as e:
        logger.warning("Insight model call failed (model=%s): %s", model, e)
        return None


def _call_model(prompt: str, model: str) -> dict[str, Any] | None:
    """Ask the model for candidate insights as JSON. Returns the parsed object, or None if it wasn't usable."""
    text = _generate(prompt, model, want_json=True)
    if not text:
        return None
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError as e:
        logger.warning("Insight model returned invalid JSON: %s", e)
        return None


def _validate_candidates(
    raw: dict[str, Any],
    evidence: list[dict[str, Any]],
    min_support: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the model's proposals into ones that are properly grounded and ones to reject (with a reason).

    The grounding check is what keeps the collection from filling with plausible-sounding inventions: a claim
    has to point at passages that were actually in front of the model.
    """
    valid_ns = {e["n"] for e in evidence}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in (raw.get("insights") or [])[: MAX_CANDIDATES_PER_CLUSTER * 2]:
        if not isinstance(item, dict):
            continue
        statement = (item.get("statement") or "").strip()
        if not statement:
            continue
        supports = [s for s in (item.get("supports") or []) if isinstance(s, int)]
        unknown = sorted(set(supports) - valid_ns)
        cited = sorted(set(supports) & valid_ns)
        if unknown:
            rejected.append({**item, "statement": statement, "decision_reason": f"cited passages that were not provided: {unknown}"})
            continue
        if len(cited) < min_support:
            rejected.append({
                **item,
                "statement": statement,
                "decision_reason": f"cited {len(cited)} passage(s); {min_support} required",
            })
            continue
        accepted.append({**item, "statement": statement, "supports": cited})
        if len(accepted) >= MAX_CANDIDATES_PER_CLUSTER:
            break
    return accepted, rejected


def _lineage_for(candidate: dict[str, Any], evidence_by_n: dict[int, dict[str, Any]], members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lineage rows for an insight: the passages it cites, the queries it came from, insights it builds on."""
    lineage = []
    for n in candidate.get("supports") or []:
        e = evidence_by_n.get(n)
        if e:
            lineage.append({
                "kind": "chunk", "relation": "supports", "group": e["group"], "source_path": e["source_path"],
                "chunk_id": e["chunk_id"], "chunk_index": e["chunk_index"], "similarity": e["similarity"],
                "text_snapshot": e["text"],
            })
    for m in members:
        lineage.append({"kind": "query", "relation": "asked", "query_id": m["id"], "text_snapshot": m["prompt"]})
    for rel, key in (("builds_on", "builds_on"), ("tensions_with", "tensions_with")):
        for iid in candidate.get(key) or []:
            if isinstance(iid, int):
                lineage.append({"kind": "insight", "relation": rel, "insight_id": iid})
    return lineage


def _decide(candidate: dict[str, Any], statement_embedding: list[float]) -> tuple[str, int | None, str | None]:
    """Create, reinforce an existing insight, or reject as already rejected before.

    Compares the candidate's own embedding to stored insight embeddings, so this works on meaning rather than
    wording and catches the same conclusion arriving in different words on a later night.
    """
    retired = find_similar_insights(
        statement_embedding, statuses=("retired", "superseded"), top_k=1, min_similarity=config.STEW_MERGE_SIMILARITY
    )
    if retired:
        r = retired[0]
        return "rejected", r["id"], f"matches retired insight {r['id']} ({r['status_reason'] or 'no reason recorded'})"
    near = find_similar_insights(
        statement_embedding, statuses=("active",), top_k=1, min_similarity=config.STEW_MERGE_SIMILARITY
    )
    if near:
        n = near[0]
        return "reinforce", n["id"], f"same conclusion as insight {n['id']} (similarity {n['similarity']})"
    return "create", None, None


def run_stew(
    *,
    write: bool = False,
    model: str | None = None,
    since: str | None = None,
    max_clusters: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one stew pass. Returns a summary dict; the reflection path is in it.

    write=False (the default) records candidates and writes the reflection without creating insights.
    """
    model = model or config.INSIGHT_MODEL
    max_clusters = max_clusters if max_clusters is not None else config.STEW_MAX_CLUSTERS
    run_id = _new_run_id()
    started = _now()
    say = progress or (lambda _m: None)

    conn = _connect_insights()
    try:
        init_stew_db(conn)
        since = since or _default_since(conn)
        conn.execute(
            "INSERT INTO stew_runs (run_id, started_at, model, dry_run, since, status) VALUES (?, ?, ?, ?, ?, 'running')",
            (run_id, started, model, int(not write), since),
        )
        conn.commit()
    finally:
        conn.close()

    summary: dict[str, Any] = {
        "run_id": run_id, "model": model, "dry_run": not write, "since": since,
        "n_queries": 0, "clusters": [], "n_created": 0, "n_reinforced": 0, "n_rejected": 0, "n_candidates": 0,
        "gaps": [], "reflection_path": None, "status": "ok", "error": None,
    }

    try:
        queries = queries_since(since)
        hits = hits_for_queries([q["id"] for q in queries])
        for q in queries:
            q["hits"] = hits.get(q["id"], [])
        summary["n_queries"] = len(queries)
        # Questions that found nothing are gaps in the collections, and belong in the reflection
        summary["gaps"] = [{"prompt": q["prompt"], "ts": q["ts"]} for q in queries if not q["result_count"]]
        say(f"{len(queries)} queries since {since}")

        clusters = _cluster_queries(queries, config.STEW_CLUSTER_THRESHOLD)
        clusters = [c for c in clusters if len(c) >= config.STEW_MIN_QUERIES]
        clusters.sort(key=len, reverse=True)
        clusters = clusters[:max_clusters]
        say(f"{len(clusters)} clusters to stew")

        for members in clusters:
            label = _cluster_label(members)
            evidence, insight_ids_seen = _gather_evidence(members, config.STEW_MAX_CHUNKS)
            cluster_summary: dict[str, Any] = {
                "label": label, "query_ids": [m["id"] for m in members], "n_queries": len(members),
                "n_chunks": len(evidence), "collections": sorted({e["group"] for e in evidence}),
                "candidates": [], "notes": None, "outcome": "no_evidence",
            }
            if not evidence:
                cluster_summary["notes"] = "no document passages were returned for these questions"
                summary["clusters"].append(cluster_summary)
                continue

            centroid = _centroid([m["embedding"] for m in members if m.get("embedding")])
            existing = find_similar_insights(centroid, statuses=("active",), top_k=MAX_EXISTING_INSIGHTS, min_similarity=0.4)
            rejected_before = find_similar_insights(
                centroid, statuses=("retired", "superseded"), top_k=MAX_REJECTED_INSIGHTS, min_similarity=0.4
            )
            say(f"cluster '{label}': {len(members)} queries, {len(evidence)} passages -> {model}")
            raw = _call_model(_format_prompt(label, members, evidence, existing, rejected_before), model)
            if raw is None:
                cluster_summary["outcome"] = "model_failed"
                cluster_summary["notes"] = "the insight model did not return usable JSON"
                summary["clusters"].append(cluster_summary)
                continue
            cluster_summary["notes"] = (raw.get("notes") or "").strip() or None

            accepted, rejected = _validate_candidates(raw, evidence, config.STEW_MIN_SUPPORT)
            evidence_by_n = {e["n"]: e for e in evidence}
            statement_embs = embed([c["statement"] for c in accepted], group=INSIGHTS_GROUP) if accepted else []

            for cand, emb in zip(accepted, statement_embs):
                decision, target_id, reason = _decide(cand, emb)
                lineage = _lineage_for(cand, evidence_by_n, members)
                insight_id = None
                if decision == "create" and write:
                    created = create_insight(
                        cand["statement"], origin="stew", actor="stew", question=(cand.get("question") or None),
                        rationale=(cand.get("rationale") or None), topic=(cand.get("topic") or None),
                        tags=cand.get("tags"), confidence=cand.get("confidence"), lineage=lineage, run_id=run_id,
                        reason=f"Synthesized from {len(members)} related questions",
                    )
                    insight_id = created["id"]
                elif decision == "reinforce" and write and target_id:
                    reinforce_insight(
                        target_id, lineage, actor="stew", run_id=run_id,
                        reason=reason or "Same conclusion reached again",
                    )
                    insight_id = target_id
                recorded = decision if write else {"create": "would_create", "reinforce": "would_reinforce"}.get(decision, decision)
                cluster_summary["candidates"].append({
                    **cand, "decision": recorded, "decision_reason": reason, "insight_id": insight_id,
                    "supporting_sources": sorted({evidence_by_n[n]["source_name"] for n in cand["supports"] if n in evidence_by_n}),
                })
                if decision == "create":
                    summary["n_created"] += 1
                elif decision == "reinforce":
                    summary["n_reinforced"] += 1
                else:
                    summary["n_rejected"] += 1

            for cand in rejected:
                cluster_summary["candidates"].append({**cand, "decision": "rejected", "insight_id": None})
                summary["n_rejected"] += 1

            summary["n_candidates"] += len(cluster_summary["candidates"])
            cluster_summary["outcome"] = "candidates" if cluster_summary["candidates"] else "nothing_worth_recording"
            summary["clusters"].append(cluster_summary)

        reflection = _write_reflection(run_id, summary, model)
        summary["reflection_path"] = str(reflection)
        _save_run(run_id, summary, finished=_now())
        action_log(
            "stew_run", group=INSIGHTS_GROUP, run_id=run_id, model=model, dry_run=not write,
            n_queries=summary["n_queries"], n_clusters=len(summary["clusters"]), n_created=summary["n_created"],
        )
        return summary
    except Exception as e:
        logger.exception("Stew run failed")
        summary["status"] = "failed"
        summary["error"] = str(e)
        _save_run(run_id, summary, finished=_now())
        raise


def _save_run(run_id: str, summary: dict[str, Any], finished: str) -> None:
    conn = _connect_insights()
    try:
        init_stew_db(conn)
        conn.execute(
            "UPDATE stew_runs SET finished_at = ?, n_queries = ?, n_clusters = ?, n_candidates = ?, n_created = ?, "
            "n_reinforced = ?, n_rejected = ?, reflection_path = ?, status = ?, error = ? WHERE run_id = ?",
            (
                finished, summary["n_queries"], len(summary["clusters"]), summary["n_candidates"],
                summary["n_created"], summary["n_reinforced"], summary["n_rejected"], summary["reflection_path"],
                summary["status"], summary["error"], run_id,
            ),
        )
        for c in summary["clusters"]:
            cur = conn.execute(
                "INSERT INTO stew_clusters (run_id, label, query_ids, n_queries, n_chunks, collections, outcome, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, c["label"], json.dumps(c["query_ids"]), c["n_queries"], c["n_chunks"],
                    json.dumps(c["collections"]), c["outcome"], c["notes"],
                ),
            )
            cluster_id = cur.lastrowid
            for cand in c["candidates"]:
                trace = {
                    k: cand.get(k) for k in
                    ("supports", "supporting_sources", "builds_on", "tensions_with", "alternatives_considered", "weak_spots")
                }
                conn.execute(
                    "INSERT INTO stew_candidates (run_id, cluster_id, statement, question, rationale, topic, tags, "
                    "confidence, trace, decision, decision_reason, insight_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id, cluster_id, cand["statement"], cand.get("question"), cand.get("rationale"),
                        cand.get("topic"), json.dumps(cand.get("tags") or []), cand.get("confidence"),
                        json.dumps(trace, ensure_ascii=False), cand.get("decision"), cand.get("decision_reason"),
                        cand.get("insight_id"), _now(),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def _reflection_prompt(summary: dict[str, Any]) -> str:
    """Ask for a narrative built from the run's own record, not from the model's memory of its thinking."""
    record = {
        "queries_considered": summary["n_queries"],
        "clusters": [
            {
                "theme": c["label"], "questions": c["n_queries"], "passages": c["n_chunks"],
                "collections": c["collections"], "outcome": c["outcome"], "model_notes": c["notes"],
                "candidates": [
                    {
                        "statement": cand["statement"], "decision": cand.get("decision"),
                        "decision_reason": cand.get("decision_reason"), "sources": cand.get("supporting_sources"),
                        "alternatives_considered": cand.get("alternatives_considered"),
                        "weak_spots": cand.get("weak_spots"), "confidence": cand.get("confidence"),
                    }
                    for cand in c["candidates"]
                ],
            }
            for c in summary["clusters"]
        ],
        "questions_that_found_nothing": [g["prompt"] for g in summary["gaps"]],
    }
    return (
        "You are writing the nightly reflection for RAGDoll, a retrieval system that builds insights from how its "
        "document collections get used. Below is the record of what tonight's run actually did.\n\n"
        "Write it up in markdown for the person who owns the collections, in these sections:\n"
        "## What I looked at\n## How the thinking went\n## What I concluded\n## What I set aside\n## Gaps\n\n"
        "Describe only what the record shows. Do not invent findings, and name the documents and themes it names. "
        "Be direct and specific; a few hundred words is plenty.\n\n"
        f"RECORD\n{json.dumps(record, indent=2, ensure_ascii=False)}"
    )


def _fallback_reflection(summary: dict[str, Any]) -> str:
    """Plain record of the run, used when the model is unavailable — better than no reflection."""
    lines = [f"# Reflection {summary['run_id']}", "", f"Model: {summary['model']}  ",
             f"Mode: {'dry run' if summary['dry_run'] else 'writing'}  ",
             f"Queries considered: {summary['n_queries']} (since {summary['since']})", ""]
    for c in summary["clusters"]:
        lines += [f"## {c['label']}", "",
                  f"{c['n_queries']} questions, {c['n_chunks']} passages, collections: {', '.join(c['collections']) or 'none'} ({c['outcome']})", ""]
        if c["notes"]:
            lines += [f"Notes: {c['notes']}", ""]
        for cand in c["candidates"]:
            lines += [f"- **{cand['decision']}**: {cand['statement']}"]
            if cand.get("decision_reason"):
                lines += [f"  - {cand['decision_reason']}"]
            if cand.get("supporting_sources"):
                lines += [f"  - sources: {', '.join(cand['supporting_sources'])}"]
        lines += [""]
    if summary["gaps"]:
        lines += ["## Gaps", "", *[f"- {g['prompt']}" for g in summary["gaps"]], ""]
    return "\n".join(lines)


def _write_reflection(run_id: str, summary: dict[str, Any], model: str) -> Path:
    path = _reflections_dir() / f"{run_id}.md"
    narrative = None
    if summary["clusters"]:
        narrative = _generate(_reflection_prompt(summary), model, want_json=False, temperature=0.4)
        if narrative is None:
            logger.warning("Reflection generation failed; writing the plain record instead")
    header = (
        f"<!-- run {run_id} | model {model} | {'dry run' if summary['dry_run'] else 'writing'} | "
        f"{summary['n_queries']} queries since {summary['since']} -->\n\n"
    )
    body = narrative or _fallback_reflection(summary)
    if narrative:
        body += "\n\n---\n\n" + _fallback_reflection(summary)
    path.write_text(header + body, encoding="utf-8")
    return path


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    conn = _connect_insights()
    try:
        init_stew_db(conn)
        rows = conn.execute("SELECT * FROM stew_runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_run(run_id: str) -> dict[str, Any] | None:
    """One run with its clusters and candidates."""
    conn = _connect_insights()
    try:
        init_stew_db(conn)
        row = conn.execute("SELECT * FROM stew_runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        run = dict(row)
        clusters = []
        for c in conn.execute("SELECT * FROM stew_clusters WHERE run_id = ? ORDER BY id", (run_id,)).fetchall():
            cluster = dict(c)
            cluster["query_ids"] = json.loads(cluster["query_ids"] or "[]")
            cluster["collections"] = json.loads(cluster["collections"] or "[]")
            cluster["candidates"] = []
            for cand in conn.execute(
                "SELECT * FROM stew_candidates WHERE cluster_id = ? ORDER BY id", (cluster["id"],)
            ).fetchall():
                d = dict(cand)
                d["tags"] = json.loads(d["tags"] or "[]")
                d["trace"] = json.loads(d["trace"] or "{}")
                cluster["candidates"].append(d)
            clusters.append(cluster)
        run["clusters"] = clusters
        return run
    finally:
        conn.close()


def read_reflection(run_id: str) -> str | None:
    path = _reflections_dir() / f"{run_id}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None

"""Garbage control: filter low-value chunks before embedding. Mandatory logging of all rejections."""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from . import config
from .action_log import log as action_log

logger = logging.getLogger(__name__)

# Common stopwords
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "may", "might", "must", "can", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "there", "then", "than", "what", "which", "who", "when", "where", "why", "how",
    "as", "if", "so", "not", "no", "yes", "up", "down", "out", "off", "over", "under", "again", "further",
    "each", "both", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than",
    "too", "very", "just", "now", "here", "there", "where", "why", "how", "all", "any", "both", "every",
}


def _count_tokens_approx(text: str) -> int:
    """Approximate token count (words * 1.3 for subword tokens)."""
    words = len(text.split())
    return int(words * 1.3)


def _lexical_diversity(text: str) -> float:
    """Calculate lexical diversity: unique words / total words."""
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", text)]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _stopword_ratio(text: str) -> float:
    """Ratio of stopwords to total words."""
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", text)]
    if not words:
        return 1.0
    stopword_count = sum(1 for w in words if w in _STOPWORDS)
    return stopword_count / len(words)


def _has_excessive_repetition(text: str) -> bool:
    """Check for excessive repetition (same word/phrase repeated many times)."""
    words = text.lower().split()
    if len(words) < 10:
        return False
    # Check if any word appears more than 50% of the time
    word_counts: dict[str, int] = {}
    for word in words:
        if len(word) >= 3:  # Ignore very short words
            word_counts[word] = word_counts.get(word, 0) + 1
    if word_counts:
        max_count = max(word_counts.values())
        return max_count > len(words) * 0.5
    return False


def _is_structural_noise(text: str) -> bool:
    """Check for structural red flags: headers-only, navigation fragments."""
    # Very short lines suggest headers/navigation
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 2 and all(len(ln) < 50 for ln in lines):
        return True
    # All caps with no lowercase suggests headers
    if len(text) > 20 and text.isupper() and not any(c.islower() for c in text):
        return True
    return False


def _check_artifact_specific(chunk: dict[str, Any]) -> tuple[bool, str]:
    """Artifact-type-specific checks. Returns (is_garbage, reason)."""
    artifact_type = chunk.get("artifact_type", "text")
    text = chunk.get("text", "")
    
    if artifact_type == "chart_summary":
        # Chart summaries should mention trends, comparisons, or data
        trend_words = {"trend", "increase", "decrease", "compare", "comparison", "data", "shows", "indicates", "higher", "lower", "peak", "decline"}
        text_lower = text.lower()
        if not any(word in text_lower for word in trend_words):
            return True, "chart_summary_no_trends"
    
    if artifact_type == "table_summary":
        # Table summaries should mention purpose, metrics, or comparisons
        purpose_words = {"table", "data", "shows", "contains", "purpose", "metric", "comparison", "ranking", "value"}
        text_lower = text.lower()
        if not any(word in text_lower for word in purpose_words):
            return True, "table_summary_no_purpose"
    
    if artifact_type == "figure_summary":
        # Figure summaries should mention steps, process, or decisions
        process_words = {"step", "process", "decision", "flow", "action", "condition", "state", "workflow"}
        text_lower = text.lower()
        if not any(word in text_lower for word in process_words):
            return True, "figure_summary_no_process"
    
    return False, ""


def _log_garbage(chunk: dict[str, Any], reason: str, stage: str, group: str, source_path: str | None = None) -> None:
    """Mandatory logging of all rejected chunks."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "reason": reason,
        "artifact_type": chunk.get("artifact_type", "text"),
        "source_path": source_path or chunk.get("source_path"),
        "page": chunk.get("page"),
        "chunk_index": chunk.get("chunk_index"),
        "text": chunk.get("text", ""),
        "text_length": len(chunk.get("text", "")),
    }
    
    # Write to garbage log file
    gp = config.get_group_paths(group)
    gp.group_dir.mkdir(parents=True, exist_ok=True)
    garbage_log = gp.group_dir / "garbage.log"
    
    with open(garbage_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    # Also log via action_log
    action_log("garbage_reject", stage=stage, reason=reason, artifact_type=chunk.get("artifact_type"), 
               text_length=len(chunk.get("text", "")), group=group)


def _llm_validate(chunk: dict[str, Any], group: str) -> bool:
    """Optional LLM-based validation: binary YES/NO on coherent, retrievable idea."""
    if not config.GARBAGE_LLM_VALIDATION:
        return True  # Skip if disabled
    
    model = config.QUERY_MODEL
    url = (config.OLLAMA_HOST or "").rstrip("/")
    text = chunk.get("text", "")
    
    prompt = (
        "Does this text express a coherent, retrievable idea that could be useful for information retrieval?\n\n"
        f"Text: {text[:500]}\n\n"
        "Respond with only YES or NO:"
    )
    
    try:
        r = requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        r.raise_for_status()
        response = r.json().get("response", "").strip().upper()
        return "YES" in response
    except Exception as e:
        logger.warning("LLM validation failed: %s, defaulting to accept", e)
        return True  # Default to accept on LLM failure


def stage1_deterministic_rejection(chunk: dict[str, Any]) -> tuple[bool, str]:
    """
    Stage 1: Fast deterministic rejection rules.
    Returns (is_garbage, reason).
    """
    text = chunk.get("text", "")
    
    # 1. Minimum length checks
    if len(text.strip()) < config.GARBAGE_MIN_CHARS:
        return True, "too_short_chars"
    
    tokens = _count_tokens_approx(text)
    if tokens < config.GARBAGE_MIN_TOKENS:
        return True, "too_short_tokens"
    
    # 2. Excessive repetition
    if _has_excessive_repetition(text):
        return True, "excessive_repetition"
    
    # 3. Very low lexical diversity
    diversity = _lexical_diversity(text)
    if diversity < config.GARBAGE_MIN_DIVERSITY:
        return True, "low_diversity"
    
    # 4. Stopword-dominant text
    stopword_ratio = _stopword_ratio(text)
    if stopword_ratio > config.GARBAGE_MAX_STOPWORD_RATIO:
        return True, "stopword_dominant"
    
    # 5. Structural red flags
    if _is_structural_noise(text):
        return True, "structural_noise"
    
    # 6. Artifact-type-specific failures
    is_garbage, reason = _check_artifact_specific(chunk)
    if is_garbage:
        return True, reason
    
    return False, ""


def stage2_deterministic_scoring(chunk: dict[str, Any]) -> float:
    """
    Stage 2: Score chunk for overall meaningfulness (0.0-1.0).
    Higher score = more meaningful.
    """
    text = chunk.get("text", "")
    if not text:
        return 0.0
    
    score = 0.0
    
    # Length signal (normalized, max 0.3)
    text_len = len(text)
    if text_len >= 200:
        score += 0.3
    elif text_len >= 100:
        score += 0.2
    elif text_len >= 50:
        score += 0.1
    
    # Lexical diversity signal (max 0.3)
    diversity = _lexical_diversity(text)
    score += diversity * 0.3
    
    # Sentence structure signal (max 0.2)
    sentences = re.split(r"[.!?]+", text)
    if len(sentences) >= 2:
        avg_sentence_len = sum(len(s.split()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
        if 5 <= avg_sentence_len <= 30:
            score += 0.2
        elif 3 <= avg_sentence_len <= 50:
            score += 0.1
    
    # Stopword ratio signal (max 0.2)
    stopword_ratio = _stopword_ratio(text)
    score += (1.0 - stopword_ratio) * 0.2
    
    return min(score, 1.0)


def filter_chunks(
    chunks: list[dict[str, Any]],
    source_path: str,
    group: str,
) -> list[dict[str, Any]]:
    """
    Filter chunks through garbage control pipeline.
    Returns only chunks that pass all stages.
    Logs all rejected chunks.
    """
    kept: list[dict[str, Any]] = []
    rejected_count = 0
    
    for chunk in chunks:
        # Stage 1: Deterministic rejection
        is_garbage, reason = stage1_deterministic_rejection(chunk)
        if is_garbage:
            _log_garbage(chunk, reason, "stage1", group, source_path)
            rejected_count += 1
            continue
        
        # Stage 2: Deterministic scoring (skip for plain text chunks to avoid dropping technical/list content)
        artifact_type = chunk.get("artifact_type", "text")
        if artifact_type != "text":
            score = stage2_deterministic_scoring(chunk)
            if score < config.GARBAGE_MIN_SCORE:
                _log_garbage(chunk, f"low_score_{score:.2f}", "stage2", group, source_path)
                rejected_count += 1
                continue
        
        # Stage 3: Optional LLM validation
        if not _llm_validate(chunk, group):
            _log_garbage(chunk, "llm_rejected", "stage3", group, source_path)
            rejected_count += 1
            continue
        
        kept.append(chunk)
    
    if rejected_count > 0:
        action_log("garbage_filtered", file=source_path, rejected=rejected_count, kept=len(kept), group=group)
        logger.info("Garbage control: rejected %d chunks, kept %d (source=%s)", rejected_count, len(kept), source_path)
    
    return kept

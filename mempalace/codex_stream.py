"""Streaming importer for large Codex CLI JSONL transcripts.

Unlike ``mine --mode convos``, this module never loads a whole transcript or
its complete chunk list into memory. It accepts the canonical Codex
``event_msg`` user/agent turns used by :mod:`mempalace.normalize`, emits
bounded exchange chunks, and upserts those chunks in fixed-size batches.

The importer owns its source revisions. A changed transcript is first
fingerprinted and counted without writing. Its new revision is then fully
upserted; only after that succeeds are earlier streaming revisions of the
same source removed. This makes an interrupted rebuild retry-safe.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .convo_miner import _detect_hall_cached, detect_convo_room
from .entities import entities_metadata
from .palace import get_collection, mine_lock


STATE_VERSION = 1
DEFAULT_MAX_CHUNKS_PER_FILE = 50_000
DRAWER_UPSERT_BATCH_SIZE = 256
INGEST_MODE = "codex_stream"


@dataclass(frozen=True)
class CodexTurn:
    role: str
    text: str
    line_no: int
    timestamp: Optional[str]


@dataclass(frozen=True)
class StreamChunk:
    chunk_index: int
    content: str
    line_start: int
    line_end: int
    authored_at: Optional[str]


@dataclass(frozen=True)
class StreamResult:
    source_file: str
    source_size: int
    source_revision: Optional[str]
    session_id: Optional[str]
    chunks_planned: int
    drawers_upserted: int
    stale_drawers_removed: int
    skipped_unchanged: bool = False
    skipped_max_chunks: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class CodexSourcePlan:
    """An immutable, verified snapshot used by batch preflight."""

    source_file: str
    source_size: int
    source_mtime_ns: int
    source_mtime: float
    source_revision: str
    session_id: str
    chunks_planned: int


BATCH_REPORT_VERSION = 1


def _configured_max_chunks(value: Optional[int]) -> int:
    if value is not None:
        if value < 0:
            raise ValueError("max_chunks_per_file must be >= 0")
        return value
    raw = os.environ.get("MEMPALACE_MAX_CHUNKS_PER_FILE")
    if raw is None:
        return DEFAULT_MAX_CHUNKS_PER_FILE
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_MAX_CHUNKS_PER_FILE
    return parsed if parsed >= 0 else DEFAULT_MAX_CHUNKS_PER_FILE


def _state_path(palace_path: str) -> Path:
    return Path(palace_path).expanduser() / ".mempalace" / "codex-stream-state.json"


def _load_state(palace_path: str) -> dict:
    path = _state_path(palace_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": STATE_VERSION, "sources": {}}
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        return {"version": STATE_VERSION, "sources": {}}
    if not isinstance(data.get("sources"), dict):
        data["sources"] = {}
    return data


def _write_state(palace_path: str, state: dict) -> None:
    path = _state_path(palace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def _source_stat(path: Path) -> tuple[int, int, float]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_mtime


def _state_matches(record: object, size: int, mtime_ns: int) -> bool:
    if not isinstance(record, dict):
        return False
    return record.get("size") == size and record.get("mtime_ns") == mtime_ns


def _source_is_unchanged(path: Path, size: int, mtime_ns: int) -> bool:
    """Return whether a source still has the snapshot used for this run."""
    try:
        current_size, current_mtime_ns, _ = _source_stat(path)
    except OSError:
        return False
    return current_size == size and current_mtime_ns == mtime_ns


def _source_path(source: str) -> Path:
    path = Path(source).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".jsonl":
        raise ValueError(f"Codex stream source must be a JSONL file: {source}")
    return path


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _session_id(path: Path) -> Optional[str]:
    """Read the Codex session ID without retaining the transcript."""
    with path.open("rb") as handle:
        for raw_line in handle:
            try:
                entry = json.loads(raw_line.decode("utf-8", "replace"))
            except ValueError:
                continue
            if not isinstance(entry, dict) or entry.get("type") != "session_meta":
                continue
            payload = entry.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("id"), str):
                return payload["id"]
            return None
    return None


def iter_codex_turns(path: Path) -> Iterator[CodexTurn]:
    """Yield only Codex's canonical event-message turns, line by line."""
    with path.open("rb") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            try:
                entry = json.loads(raw_line.decode("utf-8", "replace"))
            except ValueError:
                continue
            if not isinstance(entry, dict) or entry.get("type") != "event_msg":
                continue
            payload = entry.get("payload")
            if not isinstance(payload, dict):
                continue
            kind = payload.get("type")
            role = "user" if kind == "user_message" else "assistant" if kind == "agent_message" else None
            text = payload.get("message")
            if role is None or not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            timestamp = entry.get("timestamp")
            yield CodexTurn(
                role=role,
                text=text,
                line_no=line_no,
                timestamp=timestamp if isinstance(timestamp, str) else None,
            )


def _bounded_parts(content: str, chunk_size: int, min_chunk_size: int) -> Iterator[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if min_chunk_size < 0:
        raise ValueError("min_chunk_size must be >= 0")
    for start in range(0, len(content), chunk_size):
        part = content[start : start + chunk_size]
        if len(part.strip()) > min_chunk_size:
            yield part


def iter_codex_chunks(
    path: Path,
    *,
    chunk_size: int,
    min_chunk_size: int,
) -> Iterator[StreamChunk]:
    """Turn canonical Codex turns into bounded, ordered exchange chunks."""
    pending_user: Optional[CodexTurn] = None
    chunk_index = 0

    def emit(content: str, line_start: int, line_end: int, authored_at: Optional[str]):
        nonlocal chunk_index
        for part in _bounded_parts(content, chunk_size, min_chunk_size):
            yield StreamChunk(
                chunk_index=chunk_index,
                content=part,
                line_start=line_start,
                line_end=line_end,
                authored_at=authored_at,
            )
            chunk_index += 1

    for turn in iter_codex_turns(path):
        if turn.role == "user":
            if pending_user is not None:
                yield from emit(
                    f"> {pending_user.text}",
                    pending_user.line_no,
                    pending_user.line_no,
                    pending_user.timestamp,
                )
            pending_user = turn
            continue

        if pending_user is None:
            yield from emit(
                f"[assistant]\n{turn.text}",
                turn.line_no,
                turn.line_no,
                turn.timestamp,
            )
            continue

        yield from emit(
            f"> {pending_user.text}\n{turn.text}",
            pending_user.line_no,
            turn.line_no,
            turn.timestamp or pending_user.timestamp,
        )
        pending_user = None

    if pending_user is not None:
        yield from emit(
            f"> {pending_user.text}",
            pending_user.line_no,
            pending_user.line_no,
            pending_user.timestamp,
        )


def preflight_codex(source: str, *, chunk_size: int = 800, min_chunk_size: int = 30) -> CodexSourcePlan:
    """Read and verify one transcript without writing drawers or state.

    The returned plan is valid only while the recorded size and mtime remain
    unchanged.  Both this function and :func:`stream_codex` check that
    boundary, so a transcript that is still being written never becomes a
    completed import accidentally.
    """
    path = _source_path(source)
    source_file = str(path)
    source_size, source_mtime_ns, source_mtime = _source_stat(path)
    session_id = _session_id(path)
    if session_id is None:
        raise ValueError(f"Codex session metadata is missing from {source_file}")
    source_revision = _fingerprint(path)
    chunks_planned = sum(
        1
        for _ in iter_codex_chunks(
            path,
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
        )
    )
    if not _source_is_unchanged(path, source_size, source_mtime_ns):
        raise RuntimeError(f"Codex source changed while planning; retry later: {source_file}")
    return CodexSourcePlan(
        source_file=source_file,
        source_size=source_size,
        source_mtime_ns=source_mtime_ns,
        source_mtime=source_mtime,
        source_revision=source_revision,
        session_id=session_id,
        chunks_planned=chunks_planned,
    )


def _drawer_id(source_file: str, revision: str, chunk_index: int) -> str:
    payload = f"{source_file}\0{revision}\0{chunk_index}".encode("utf-8")
    return f"codex_stream_{hashlib.sha256(payload).hexdigest()}"


def _delete_prior_revisions(collection, source_file: str, revision: str) -> int:
    """Remove only completed older streaming revisions of one source."""
    offset = 0
    stale_ids: list[str] = []
    while True:
        batch = collection.get(
            where={"source_file": source_file},
            limit=1000,
            offset=offset,
            include=["metadatas"],
        )
        ids = batch.get("ids") or []
        metadatas = batch.get("metadatas") or []
        if not ids:
            break
        for drawer_id, metadata in zip(ids, metadatas):
            if (
                isinstance(metadata, dict)
                and metadata.get("ingest_mode") == INGEST_MODE
                and metadata.get("source_revision") != revision
            ):
                stale_ids.append(drawer_id)
        offset += len(ids)

    for start in range(0, len(stale_ids), DRAWER_UPSERT_BATCH_SIZE):
        collection.delete(ids=stale_ids[start : start + DRAWER_UPSERT_BATCH_SIZE])
    return len(stale_ids)


def _upsert_revision(
    collection,
    *,
    source_file: str,
    source_revision: str,
    source_mtime: float,
    source_size: int,
    session_id: Optional[str],
    wing: str,
    agent: str,
    chunks: Iterator[StreamChunk],
) -> int:
    filed_at = datetime.now().isoformat()
    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_metas: list[dict] = []
    upserted = 0

    def flush() -> None:
        nonlocal upserted
        if not batch_ids:
            return
        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
        upserted += len(batch_ids)
        batch_ids.clear()
        batch_docs.clear()
        batch_metas.clear()

    for chunk in chunks:
        room = detect_convo_room(chunk.content)
        batch_ids.append(_drawer_id(source_file, source_revision, chunk.chunk_index))
        batch_docs.append(chunk.content)
        batch_metas.append(
            {
                "wing": wing,
                "room": room,
                "hall": _detect_hall_cached(chunk.content),
                "entities": entities_metadata(chunk.content),
                "source_file": source_file,
                "source_revision": source_revision,
                "source_mtime": source_mtime,
                "source_size": source_size,
                "session_id": session_id or "unknown",
                "chunk_index": chunk.chunk_index,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "authored_at": chunk.authored_at or filed_at,
                "added_by": agent,
                "filed_at": filed_at,
                "ingest_mode": INGEST_MODE,
            }
        )
        if len(batch_ids) >= DRAWER_UPSERT_BATCH_SIZE:
            flush()
    flush()
    return upserted


def stream_codex(
    source: str,
    palace_path: str,
    *,
    wing: str = "codex_stream",
    agent: str = "mempalace",
    dry_run: bool = False,
    max_chunks_per_file: Optional[int] = None,
    chunk_size: int = 800,
    min_chunk_size: int = 30,
) -> StreamResult:
    """Import one Codex JSONL transcript without whole-file buffering."""
    path = _source_path(source)
    source_file = str(path)
    source_size, source_mtime_ns, source_mtime = _source_stat(path)
    state = _load_state(palace_path)
    existing = state["sources"].get(source_file)
    if _state_matches(existing, source_size, source_mtime_ns):
        return StreamResult(
            source_file=source_file,
            source_size=source_size,
            source_revision=existing.get("revision"),
            session_id=existing.get("session_id"),
            chunks_planned=existing.get("chunks", 0),
            drawers_upserted=0,
            stale_drawers_removed=0,
            skipped_unchanged=True,
            dry_run=dry_run,
        )

    plan = preflight_codex(
        source_file,
        chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
    )
    source_size = plan.source_size
    source_mtime_ns = plan.source_mtime_ns
    source_mtime = plan.source_mtime
    session_id = plan.session_id
    source_revision = plan.source_revision
    chunks_planned = plan.chunks_planned
    maximum = _configured_max_chunks(max_chunks_per_file)
    if maximum and chunks_planned > maximum:
        return StreamResult(
            source_file=source_file,
            source_size=source_size,
            source_revision=source_revision,
            session_id=session_id,
            chunks_planned=chunks_planned,
            drawers_upserted=0,
            stale_drawers_removed=0,
            skipped_max_chunks=True,
            dry_run=dry_run,
        )
    if dry_run:
        return StreamResult(
            source_file=source_file,
            source_size=source_size,
            source_revision=source_revision,
            session_id=session_id,
            chunks_planned=chunks_planned,
            drawers_upserted=0,
            stale_drawers_removed=0,
            dry_run=True,
        )

    with mine_lock(source_file):
        # A concurrent importer may have committed while this process was
        # fingerprinting. Re-read the state under the source lock.
        state = _load_state(palace_path)
        existing = state["sources"].get(source_file)
        if _state_matches(existing, source_size, source_mtime_ns):
            return StreamResult(
                source_file=source_file,
                source_size=source_size,
                source_revision=existing.get("revision"),
                session_id=existing.get("session_id"),
                chunks_planned=existing.get("chunks", 0),
                drawers_upserted=0,
                stale_drawers_removed=0,
                skipped_unchanged=True,
            )
        if not _source_is_unchanged(path, source_size, source_mtime_ns):
            raise RuntimeError(f"Codex source changed before import; retry later: {source_file}")

        collection = get_collection(palace_path, create=True)
        upserted = _upsert_revision(
            collection,
            source_file=source_file,
            source_revision=source_revision,
            source_mtime=source_mtime,
            source_size=source_size,
            session_id=session_id,
            wing=wing,
            agent=agent,
            chunks=iter_codex_chunks(
                path,
                chunk_size=chunk_size,
                min_chunk_size=min_chunk_size,
            ),
        )
        if not _source_is_unchanged(path, source_size, source_mtime_ns):
            # Leave the deterministic, partially-or-fully written revision in
            # place. A later successful revision will upsert its own IDs and
            # remove this one; do not make a stale source look complete.
            raise RuntimeError(f"Codex source changed during import; retry later: {source_file}")
        removed = _delete_prior_revisions(collection, source_file, source_revision)
        state["sources"][source_file] = {
            "revision": source_revision,
            "size": source_size,
            "mtime_ns": source_mtime_ns,
            "session_id": session_id,
            "chunks": chunks_planned,
        }
        _write_state(palace_path, state)

    return StreamResult(
        source_file=source_file,
        source_size=source_size,
        source_revision=source_revision,
        session_id=session_id,
        chunks_planned=chunks_planned,
        drawers_upserted=upserted,
        stale_drawers_removed=removed,
    )


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def _batch_entry_from_plan(plan: CodexSourcePlan, status: str) -> dict:
    return {
        "source_file": plan.source_file,
        "source_size": plan.source_size,
        "source_mtime_ns": plan.source_mtime_ns,
        "source_mtime": plan.source_mtime,
        "source_revision": plan.source_revision,
        "session_id": plan.session_id,
        "chunks_planned": plan.chunks_planned,
        "status": status,
    }


def _update_batch_summary(report: dict) -> None:
    entries = report.get("entries") or []
    status_counts: dict[str, int] = {}
    planned_chunks = 0
    source_bytes = 0
    for entry in entries:
        status = entry.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        planned_chunks += int(entry.get("chunks_planned") or 0)
        source_bytes += int(entry.get("source_size") or 0)
    report["summary"] = {
        "entries": len(entries),
        "status_counts": status_counts,
        "planned_chunks": planned_chunks,
        "source_bytes": source_bytes,
    }


def _batch_error_count(report: dict) -> int:
    errors = {
        "preflight_error",
        "duplicate_conflict",
        "blocked_oversized_cap",
        "changed_since_preflight",
        "failed",
        "blocked_by_cap_during_execute",
    }
    return sum(1 for entry in report.get("entries") or [] if entry.get("status") in errors)


def _load_batch_report(path: Path) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read batch manifest {path}: {exc}") from exc
    if not isinstance(report, dict) or report.get("version") != BATCH_REPORT_VERSION:
        raise ValueError(f"Unsupported batch manifest version: {path}")
    if not isinstance(report.get("entries"), list) or not isinstance(report.get("root"), str):
        raise ValueError(f"Invalid batch manifest: {path}")
    return report


def _assert_batch_caps(normal_max_chunks: Optional[int], oversized_max_chunks: int) -> tuple[int, int]:
    normal = _configured_max_chunks(normal_max_chunks)
    if normal <= 0:
        raise ValueError("codex-stream-batch requires a positive normal chunk cap")
    if oversized_max_chunks < normal:
        raise ValueError("oversized_max_chunks must be at least the normal chunk cap")
    return normal, oversized_max_chunks


def preflight_codex_batch(
    root: str,
    report_path: str,
    *,
    stability_minutes: int = 60,
    normal_max_chunks: Optional[int] = None,
    oversized_max_chunks: int = 125_000,
    chunk_size: int = 800,
    min_chunk_size: int = 30,
) -> dict:
    """Create a no-write manifest for a stable Codex transcript tree.

    The manifest is deliberately immutable input for execution.  It records
    the exact stat snapshot, content revision and chunk count so a later run
    can skip anything that changed rather than importing a moving target.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"Codex batch root must be a directory: {root}")
    if stability_minutes < 0:
        raise ValueError("stability_minutes must be >= 0")
    normal_cap, oversized_cap = _assert_batch_caps(normal_max_chunks, oversized_max_chunks)
    manifest_path = Path(report_path).expanduser().resolve()
    if manifest_path.exists():
        raise ValueError(f"Refusing to overwrite existing batch manifest: {manifest_path}")

    report: dict = {
        "version": BATCH_REPORT_VERSION,
        "created_at": datetime.now().isoformat(),
        "root": str(root_path),
        "stability_minutes": stability_minutes,
        "normal_max_chunks_per_file": normal_cap,
        "oversized_max_chunks_per_file": oversized_cap,
        "entries": [],
    }
    entries: list[dict] = report["entries"]
    seen_sessions: dict[str, dict] = {}
    cutoff_seconds = stability_minutes * 60
    now = time.time()

    sources = sorted(path for path in root_path.rglob("*.jsonl") if path.is_file())
    for candidate in sources:
        try:
            path = candidate.resolve()
            path.relative_to(root_path)
            source_size, source_mtime_ns, source_mtime = _source_stat(path)
        except (OSError, ValueError) as exc:
            entries.append({"source_file": str(candidate), "status": "preflight_error", "error": str(exc)})
            continue

        age_seconds = max(0, now - source_mtime)
        if age_seconds < cutoff_seconds:
            entries.append(
                {
                    "source_file": str(path),
                    "source_size": source_size,
                    "source_mtime_ns": source_mtime_ns,
                    "source_mtime": source_mtime,
                    "age_seconds": age_seconds,
                    "status": "unstable",
                }
            )
            continue

        try:
            plan = preflight_codex(
                str(path),
                chunk_size=chunk_size,
                min_chunk_size=min_chunk_size,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            entries.append(
                {
                    "source_file": str(path),
                    "source_size": source_size,
                    "source_mtime_ns": source_mtime_ns,
                    "source_mtime": source_mtime,
                    "status": "preflight_error",
                    "error": str(exc),
                }
            )
            continue

        if plan.chunks_planned <= normal_cap:
            status = "ready"
        elif plan.chunks_planned <= oversized_cap:
            status = "requires_oversized_approval"
        else:
            status = "blocked_oversized_cap"
        entry = _batch_entry_from_plan(plan, status)

        prior = seen_sessions.get(plan.session_id)
        if prior is None:
            seen_sessions[plan.session_id] = entry
        elif prior.get("source_revision") == plan.source_revision:
            entry["status"] = "duplicate_identical"
            entry["duplicate_of"] = prior["source_file"]
        else:
            entry["status"] = "duplicate_conflict"
            entry["conflict_with"] = prior["source_file"]
            prior["status"] = "duplicate_conflict"
            prior["conflict_with"] = entry["source_file"]
        entries.append(entry)

    _update_batch_summary(report)
    _write_json_atomic(manifest_path, report)
    return report


def _stream_result_data(result: StreamResult) -> dict:
    return {
        "source_revision": result.source_revision,
        "chunks_planned": result.chunks_planned,
        "drawers_upserted": result.drawers_upserted,
        "stale_drawers_removed": result.stale_drawers_removed,
        "skipped_unchanged": result.skipped_unchanged,
        "skipped_max_chunks": result.skipped_max_chunks,
    }


def execute_codex_batch(
    report_path: str,
    palace_path: str,
    *,
    wing: str = "codex_archive",
    agent: str = "mempalace",
    approve_oversized_session_ids: Optional[list[str]] = None,
    chunk_size: int = 800,
    min_chunk_size: int = 30,
) -> dict:
    """Resume a preflighted batch, persisting progress after every source."""
    manifest_path = Path(report_path).expanduser().resolve()
    report = _load_batch_report(manifest_path)
    root_path = Path(report["root"]).resolve()
    normal_cap, oversized_cap = _assert_batch_caps(
        report.get("normal_max_chunks_per_file"),
        int(report.get("oversized_max_chunks_per_file") or 0),
    )
    approved = set(approve_oversized_session_ids or [])
    entries: list[dict] = report["entries"]
    oversized_sessions = {
        entry.get("session_id")
        for entry in entries
        if isinstance(entry.get("session_id"), str)
        and int(entry.get("chunks_planned") or 0) > normal_cap
        and int(entry.get("chunks_planned") or 0) <= oversized_cap
    }
    unknown_approvals = approved - oversized_sessions
    if unknown_approvals:
        raise ValueError(
            "Oversized approval does not match a preflighted eligible session: "
            + ", ".join(sorted(unknown_approvals))
        )

    execution = report.setdefault("execution", {"attempts": []})
    attempts = execution.setdefault("attempts", [])
    attempt = {
        "started_at": datetime.now().isoformat(),
        "approved_oversized_session_ids": sorted(approved),
    }
    attempts.append(attempt)
    for entry in entries:
        if entry.get("status") == "requires_oversized_approval" and entry.get("session_id") in approved:
            entry["status"] = "approved_oversized"
            entry["approved_at"] = datetime.now().isoformat()
    _update_batch_summary(report)
    report["updated_at"] = datetime.now().isoformat()
    _write_json_atomic(manifest_path, report)

    attempted = 0
    errors = 0
    for entry in entries:
        if entry.get("status") not in {"ready", "approved_oversized", "failed"}:
            continue
        source_file = entry.get("source_file")
        if not isinstance(source_file, str):
            entry["status"] = "failed"
            entry["error"] = "manifest source_file is missing"
            errors += 1
        else:
            attempted += 1
            try:
                path = _source_path(source_file)
                path.relative_to(root_path)
                expected_size = int(entry["source_size"])
                expected_mtime_ns = int(entry["source_mtime_ns"])
                if not _source_is_unchanged(path, expected_size, expected_mtime_ns):
                    entry["status"] = "changed_since_preflight"
                    entry["error"] = "source stat snapshot no longer matches manifest; preflight again"
                    errors += 1
                else:
                    chunks_planned = int(entry["chunks_planned"])
                    max_chunks = oversized_cap if chunks_planned > normal_cap else normal_cap
                    result = stream_codex(
                        str(path),
                        palace_path,
                        wing=wing,
                        agent=agent,
                        max_chunks_per_file=max_chunks,
                        chunk_size=chunk_size,
                        min_chunk_size=min_chunk_size,
                    )
                    entry["result"] = _stream_result_data(result)
                    entry["completed_at"] = datetime.now().isoformat()
                    if result.skipped_max_chunks:
                        entry["status"] = "blocked_by_cap_during_execute"
                        entry["error"] = "current source exceeded its approved chunk cap"
                        errors += 1
                    elif result.skipped_unchanged:
                        entry["status"] = "unchanged"
                    else:
                        entry["status"] = "imported"
                        entry.pop("error", None)
            except (OSError, ValueError, RuntimeError) as exc:
                entry["status"] = "failed"
                entry["error"] = str(exc)
                errors += 1

        _update_batch_summary(report)
        report["updated_at"] = datetime.now().isoformat()
        _write_json_atomic(manifest_path, report)

    attempt["finished_at"] = datetime.now().isoformat()
    attempt["sources_attempted"] = attempted
    attempt["errors"] = errors
    _update_batch_summary(report)
    report["updated_at"] = datetime.now().isoformat()
    _write_json_atomic(manifest_path, report)
    return report

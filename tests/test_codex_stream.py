"""Tests for the bounded Codex JSONL streaming importer."""

from __future__ import annotations

import json
from contextlib import contextmanager

import mempalace.codex_stream as codex_stream


class FakeCollection:
    def __init__(self):
        self.rows = {}

    def upsert(self, *, ids, documents, metadatas):
        for drawer_id, document, metadata in zip(ids, documents, metadatas):
            self.rows[drawer_id] = {"document": document, "metadata": metadata}

    def get(self, *, ids=None, where=None, limit=None, offset=0, include=None):
        if ids is not None:
            selected = [(drawer_id, self.rows[drawer_id]) for drawer_id in ids if drawer_id in self.rows]
        else:
            selected = list(self.rows.items())
            if where and "source_file" in where:
                selected = [
                    (drawer_id, row)
                    for drawer_id, row in selected
                    if row["metadata"].get("source_file") == where["source_file"]
                ]
            selected = selected[offset : offset + (limit or len(selected))]
        return {
            "ids": [drawer_id for drawer_id, _ in selected],
            "metadatas": [row["metadata"] for _, row in selected],
        }

    def delete(self, *, ids):
        for drawer_id in ids:
            self.rows.pop(drawer_id, None)


@contextmanager
def no_lock(_source_file):
    yield


def write_codex(path, *, answer="Use bounded batches for large files.", session_id="session-123"):
    records = [
        {"type": "session_meta", "payload": {"id": session_id, "cwd": "/repo/demo"}},
        {
            "type": "event_msg",
            "timestamp": "2026-07-28T00:00:00Z",
            "payload": {"type": "user_message", "message": "How should we ingest large sessions?"},
        },
        {
            "type": "event_msg",
            "timestamp": "2026-07-28T00:00:01Z",
            "payload": {"type": "agent_message", "message": answer},
        },
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "developer", "content": []},
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_iter_codex_chunks_uses_only_canonical_event_messages(tmp_path):
    source = tmp_path / "session.jsonl"
    write_codex(source, answer="A" * 120)

    chunks = list(codex_stream.iter_codex_chunks(source, chunk_size=50, min_chunk_size=0))

    assert len(chunks) >= 3
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(len(chunk.content) <= 50 for chunk in chunks)
    assert chunks[0].content.startswith("> How should we ingest large sessions?")
    assert all(chunk.line_start == 2 and chunk.line_end == 3 for chunk in chunks)


def test_stream_rebuilds_changed_source_after_successful_new_revision(tmp_path, monkeypatch):
    source = tmp_path / "session.jsonl"
    palace = tmp_path / "palace"
    write_codex(source, answer="A" * 120)
    collection = FakeCollection()
    monkeypatch.setattr(codex_stream, "get_collection", lambda *args, **kwargs: collection)
    monkeypatch.setattr(codex_stream, "mine_lock", no_lock)

    first = codex_stream.stream_codex(
        str(source), str(palace), wing="benchmark", chunk_size=50, min_chunk_size=0
    )
    assert first.drawers_upserted >= 3
    assert len(collection.rows) == first.drawers_upserted

    unchanged = codex_stream.stream_codex(
        str(source), str(palace), wing="benchmark", chunk_size=50, min_chunk_size=0
    )
    assert unchanged.skipped_unchanged
    assert len(collection.rows) == first.drawers_upserted

    write_codex(source, answer="B" * 160)
    changed = codex_stream.stream_codex(
        str(source), str(palace), wing="benchmark", chunk_size=50, min_chunk_size=0
    )
    assert changed.drawers_upserted > 0
    assert changed.stale_drawers_removed == first.drawers_upserted
    assert len(collection.rows) == changed.drawers_upserted
    assert {row["metadata"]["source_revision"] for row in collection.rows.values()} == {
        changed.source_revision
    }


def test_stream_indexes_session_workspace_as_provenance_and_context(tmp_path, monkeypatch):
    source = tmp_path / "session.jsonl"
    palace = tmp_path / "palace"
    write_codex(source)
    collection = FakeCollection()
    monkeypatch.setattr(codex_stream, "get_collection", lambda *args, **kwargs: collection)
    monkeypatch.setattr(codex_stream, "mine_lock", no_lock)

    result = codex_stream.stream_codex(str(source), str(palace))

    assert result.session_cwd == "/repo/demo"
    assert collection.rows
    assert all(row["metadata"]["session_cwd"] == "/repo/demo" for row in collection.rows.values())
    assert all(
        row["document"].startswith("[Codex workspace: /repo/demo]\n")
        for row in collection.rows.values()
    )
    state = json.loads((palace / ".mempalace" / "codex-stream-state.json").read_text())
    assert state["version"] == codex_stream.STATE_VERSION
    assert state["sources"][str(source.resolve())]["session_cwd"] == "/repo/demo"


def test_stream_uses_logical_source_id_for_stable_provenance(tmp_path, monkeypatch):
    source = tmp_path / "materialized.jsonl"
    palace = tmp_path / "palace"
    source_id = "bookkeeper://record/018f2caa-8fa7-7a65-b9d9-9f4a6a9e2818"
    write_codex(source)
    collection = FakeCollection()
    monkeypatch.setattr(codex_stream, "get_collection", lambda *args, **kwargs: collection)
    monkeypatch.setattr(codex_stream, "mine_lock", no_lock)

    result = codex_stream.stream_codex(str(source), str(palace), source_id=source_id)

    assert result.source_file == source_id
    assert all(row["metadata"]["source_file"] == source_id for row in collection.rows.values())
    state = json.loads((palace / ".mempalace" / "codex-stream-state.json").read_text())
    assert source_id in state["sources"]
    assert str(source.resolve()) not in state["sources"]


def test_stream_reimports_when_the_index_schema_state_advances(tmp_path, monkeypatch):
    source = tmp_path / "session.jsonl"
    palace = tmp_path / "palace"
    write_codex(source)
    collection = FakeCollection()
    monkeypatch.setattr(codex_stream, "get_collection", lambda *_args, **_kwargs: collection)

    first = codex_stream.stream_codex(str(source), str(palace), chunk_size=50, min_chunk_size=0)
    assert first.drawers_upserted > 0

    state_path = palace / ".mempalace" / "codex-stream-state.json"
    state = json.loads(state_path.read_text())
    state["version"] = codex_stream.STATE_VERSION - 1
    state_path.write_text(json.dumps(state))

    refreshed = codex_stream.stream_codex(
        str(source), str(palace), chunk_size=50, min_chunk_size=0
    )
    assert refreshed.drawers_upserted == first.drawers_upserted
    assert not refreshed.skipped_unchanged


def test_stream_dry_run_and_chunk_cap_do_not_write_state_or_drawers(tmp_path, monkeypatch):
    source = tmp_path / "session.jsonl"
    palace = tmp_path / "palace"
    write_codex(source, answer="C" * 120)
    collection = FakeCollection()
    monkeypatch.setattr(codex_stream, "get_collection", lambda *args, **kwargs: collection)
    monkeypatch.setattr(codex_stream, "mine_lock", no_lock)

    dry_run = codex_stream.stream_codex(
        str(source), str(palace), dry_run=True, chunk_size=50, min_chunk_size=0
    )
    assert dry_run.dry_run
    assert dry_run.chunks_planned >= 3
    assert not collection.rows
    assert not (palace / ".mempalace" / "codex-stream-state.json").exists()

    capped = codex_stream.stream_codex(
        str(source), str(palace), max_chunks_per_file=1, chunk_size=50, min_chunk_size=0
    )
    assert capped.skipped_max_chunks
    assert not collection.rows
    assert not (palace / ".mempalace" / "codex-stream-state.json").exists()


def test_stream_never_marks_a_source_complete_if_it_changes_during_upsert(tmp_path, monkeypatch):
    source = tmp_path / "session.jsonl"
    palace = tmp_path / "palace"
    write_codex(source, answer="D" * 120)
    collection = FakeCollection()
    monkeypatch.setattr(codex_stream, "get_collection", lambda *args, **kwargs: collection)
    monkeypatch.setattr(codex_stream, "mine_lock", no_lock)
    unchanged_checks = iter((True, True, False))
    monkeypatch.setattr(
        codex_stream,
        "_source_is_unchanged",
        lambda *args, **kwargs: next(unchanged_checks),
    )

    try:
        codex_stream.stream_codex(
            str(source), str(palace), wing="benchmark", chunk_size=50, min_chunk_size=0
        )
    except RuntimeError as exc:
        assert "changed during import" in str(exc)
    else:
        assert False, "a changing source must not be marked complete"

    assert collection.rows
    assert not (palace / ".mempalace" / "codex-stream-state.json").exists()


def test_batch_preflight_and_execution_require_explicit_oversized_approval(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    root.mkdir()
    small = root / "small.jsonl"
    large = root / "large.jsonl"
    report_path = tmp_path / "batch.json"
    palace = tmp_path / "palace"
    write_codex(small, answer="small", session_id="session-small")
    write_codex(large, answer="L" * 400, session_id="session-large")
    collection = FakeCollection()
    monkeypatch.setattr(codex_stream, "get_collection", lambda *args, **kwargs: collection)
    monkeypatch.setattr(codex_stream, "mine_lock", no_lock)

    report = codex_stream.preflight_codex_batch(
        str(root),
        str(report_path),
        stability_minutes=0,
        normal_max_chunks=2,
        oversized_max_chunks=20,
        chunk_size=50,
        min_chunk_size=0,
    )
    assert report["summary"]["status_counts"]["ready"] == 1
    assert report["summary"]["status_counts"]["requires_oversized_approval"] == 1
    assert report["entries"][0]["session_cwd"] == "/repo/demo"
    assert not collection.rows

    first = codex_stream.execute_codex_batch(
        str(report_path), str(palace), chunk_size=50, min_chunk_size=0
    )
    assert first["summary"]["status_counts"]["imported"] == 1
    assert first["summary"]["status_counts"]["requires_oversized_approval"] == 1

    completed = codex_stream.execute_codex_batch(
        str(report_path),
        str(palace),
        approve_oversized_session_ids=["session-large"],
        chunk_size=50,
        min_chunk_size=0,
    )
    assert completed["summary"]["status_counts"]["imported"] == 2
    assert len(collection.rows) > 2


def test_batch_execution_skips_sources_changed_since_preflight(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    root.mkdir()
    source = root / "session.jsonl"
    report_path = tmp_path / "batch.json"
    palace = tmp_path / "palace"
    write_codex(source, session_id="session-changing")
    collection = FakeCollection()
    monkeypatch.setattr(codex_stream, "get_collection", lambda *args, **kwargs: collection)
    monkeypatch.setattr(codex_stream, "mine_lock", no_lock)

    codex_stream.preflight_codex_batch(
        str(root),
        str(report_path),
        stability_minutes=0,
        normal_max_chunks=20,
        chunk_size=50,
        min_chunk_size=0,
    )
    write_codex(source, answer="changed after preflight", session_id="session-changing")
    report = codex_stream.execute_codex_batch(
        str(report_path), str(palace), chunk_size=50, min_chunk_size=0
    )
    assert report["summary"]["status_counts"]["changed_since_preflight"] == 1
    assert not collection.rows

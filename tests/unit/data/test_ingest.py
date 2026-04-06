# tests/unit/data/test_ingest.py
"""
Unit tests for app/data/ingest.py.

parse_misra_file: uses pytest tmp_path for real file I/O.
upload_to_mongodb: uses AsyncMock to avoid any real DB connection.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.data.ingest import parse_misra_file, upload_to_mongodb, main, run_ingest_cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_rules_file(tmp_path: Path, content: str) -> Path:
    """Write content to a temp file and return a path relative to project root
    that parse_misra_file can resolve via its base_dir logic.

    parse_misra_file resolves: base_dir / filepath
    where base_dir = project root (parent of app/).
    We write to tmp_path but patch Path.resolve so base_dir becomes tmp_path.
    """
    rules_file = tmp_path / "rules.txt"
    rules_file.write_text(content, encoding="utf-8")
    return rules_file


def _make_mongodb_service(bulk_result_counts: tuple[int, int] = (5, 0)) -> MagicMock:
    svc = MagicMock()
    bulk_result = MagicMock()
    bulk_result.upserted_count = bulk_result_counts[0]
    bulk_result.modified_count = bulk_result_counts[1]
    svc.collection = MagicMock()
    svc.collection.bulk_write = AsyncMock(return_value=bulk_result)
    svc.create_indexes = AsyncMock(return_value=None)
    return svc


# ---------------------------------------------------------------------------
# parse_misra_file — happy paths
# ---------------------------------------------------------------------------

def test_parse_misra_parses_rule_header(tmp_path: Path):
    (tmp_path / "rules.txt").write_text(
        "Rule 1.1    Required\nNo dead code allowed.\n",
        encoding="utf-8",
    )
    with patch("app.data.ingest.Path") as mock_path_cls:
        # Make base_dir / filepath resolve to our tmp file
        mock_path_cls.return_value.resolve.return_value.parent.parent.parent = tmp_path
        mock_path_cls.return_value.resolve.return_value = Path(__file__)  # ignored
        # Simpler: just patch open directly
        pass

    # Use the real file system: patch base_dir inside the function
    import app.data.ingest as ingest_mod
    original_path = ingest_mod.Path

    class _FakePath:
        def __init__(self, *args):
            self._p = Path(*args)
        def resolve(self):
            return self
        @property
        def parent(self):
            # Return a FakePath whose / operator points into tmp_path
            return _FakeParent(tmp_path)
        def __truediv__(self, other):
            return self._p / other
        def __str__(self):
            return str(self._p)

    class _FakeParent:
        def __init__(self, root):
            self._root = root
        @property
        def parent(self):
            return self
        def __truediv__(self, other):
            return self._root / other

    with patch.object(ingest_mod, "Path", side_effect=lambda *a: _FakePath(*a)):
        # Write the file at the resolved location
        target = tmp_path / "rules.txt"
        target.write_text("Rule 1.1    Required\nNo dead code allowed.\n", encoding="utf-8")
        rules = parse_misra_file("rules.txt")

    assert len(rules) == 1
    assert rules[0]["rule_type"] == "RULE"
    assert rules[0]["section"] == 1
    assert rules[0]["rule_number"] == 1
    assert rules[0]["category"] == "Required"
    assert rules[0]["full_text"] == "No dead code allowed."


def test_parse_misra_parses_dir_header(tmp_path: Path):
    import app.data.ingest as ingest_mod

    class _FakeParent:
        def __init__(self, root): self._root = root
        @property
        def parent(self): return self
        def __truediv__(self, other): return self._root / other

    class _FakePath:
        def __init__(self, *a): self._p = Path(*a)
        def resolve(self): return self
        @property
        def parent(self): return _FakeParent(tmp_path)
        def __truediv__(self, other): return self._p / other

    (tmp_path / "rules.txt").write_text(
        "Dir 4.1\tRequired\nRun-time failures shall be minimized.\n",
        encoding="utf-8",
    )
    with patch.object(ingest_mod, "Path", side_effect=lambda *a: _FakePath(*a)):
        rules = parse_misra_file("rules.txt")

    assert rules[0]["rule_type"] == "DIR"
    assert rules[0]["section"] == 4
    assert rules[0]["rule_number"] == 1


def test_parse_misra_skips_comments_and_blank_lines(tmp_path: Path):
    import app.data.ingest as ingest_mod

    class _FakeParent:
        def __init__(self, root): self._root = root
        @property
        def parent(self): return self
        def __truediv__(self, other): return self._root / other

    class _FakePath:
        def __init__(self, *a): self._p = Path(*a)
        def resolve(self): return self
        @property
        def parent(self): return _FakeParent(tmp_path)
        def __truediv__(self, other): return self._p / other

    content = (
        "# This is a comment\n"
        "\n"
        "Rule 2.2    Advisory\n"
        "Feasibility rule.\n"
    )
    (tmp_path / "rules.txt").write_text(content, encoding="utf-8")
    with patch.object(ingest_mod, "Path", side_effect=lambda *a: _FakePath(*a)):
        rules = parse_misra_file("rules.txt")

    assert len(rules) == 1
    assert rules[0]["full_text"] == "Feasibility rule."


def test_parse_misra_multiline_text_concatenated_with_space(tmp_path: Path):
    import app.data.ingest as ingest_mod

    class _FakeParent:
        def __init__(self, root): self._root = root
        @property
        def parent(self): return self
        def __truediv__(self, other): return self._root / other

    class _FakePath:
        def __init__(self, *a): self._p = Path(*a)
        def resolve(self): return self
        @property
        def parent(self): return _FakeParent(tmp_path)
        def __truediv__(self, other): return self._p / other

    content = (
        "Rule 3.1    Required\n"
        "First line of text.\n"
        "Second line of text.\n"
    )
    (tmp_path / "rules.txt").write_text(content, encoding="utf-8")
    with patch.object(ingest_mod, "Path", side_effect=lambda *a: _FakePath(*a)):
        rules = parse_misra_file("rules.txt")

    assert rules[0]["full_text"] == "First line of text. Second line of text."


def test_parse_misra_last_rule_captured(tmp_path: Path):
    """The final rule in the file must be included even without a trailing newline."""
    import app.data.ingest as ingest_mod

    class _FakeParent:
        def __init__(self, root): self._root = root
        @property
        def parent(self): return self
        def __truediv__(self, other): return self._root / other

    class _FakePath:
        def __init__(self, *a): self._p = Path(*a)
        def resolve(self): return self
        @property
        def parent(self): return _FakeParent(tmp_path)
        def __truediv__(self, other): return self._p / other

    content = "Rule 1.1    Required\nFirst rule.\nRule 2.2    Advisory\nLast rule."
    (tmp_path / "rules.txt").write_text(content, encoding="utf-8")
    with patch.object(ingest_mod, "Path", side_effect=lambda *a: _FakePath(*a)):
        rules = parse_misra_file("rules.txt")

    assert len(rules) == 2
    assert rules[-1]["full_text"] == "Last rule."


def test_parse_misra_empty_file_returns_empty_list(tmp_path: Path):
    import app.data.ingest as ingest_mod

    class _FakeParent:
        def __init__(self, root): self._root = root
        @property
        def parent(self): return self
        def __truediv__(self, other): return self._root / other

    class _FakePath:
        def __init__(self, *a): self._p = Path(*a)
        def resolve(self): return self
        @property
        def parent(self): return _FakeParent(tmp_path)
        def __truediv__(self, other): return self._p / other

    (tmp_path / "rules.txt").write_text("", encoding="utf-8")
    with patch.object(ingest_mod, "Path", side_effect=lambda *a: _FakePath(*a)):
        rules = parse_misra_file("rules.txt")

    assert rules == []


def test_parse_misra_file_not_found_returns_empty_list(tmp_path: Path):
    import app.data.ingest as ingest_mod

    class _FakeParent:
        def __init__(self, root): self._root = root
        @property
        def parent(self): return self
        def __truediv__(self, other): return self._root / other

    class _FakePath:
        def __init__(self, *a): self._p = Path(*a)
        def resolve(self): return self
        @property
        def parent(self): return _FakeParent(tmp_path)
        def __truediv__(self, other): return self._p / other

    with patch.object(ingest_mod, "Path", side_effect=lambda *a: _FakePath(*a)):
        rules = parse_misra_file("nonexistent_file.txt")

    assert rules == []


def test_parse_misra_tab_separated_header(tmp_path: Path):
    """Rule 22.15\tMandatory — tab separator must be handled."""
    import app.data.ingest as ingest_mod

    class _FakeParent:
        def __init__(self, root): self._root = root
        @property
        def parent(self): return self
        def __truediv__(self, other): return self._root / other

    class _FakePath:
        def __init__(self, *a): self._p = Path(*a)
        def resolve(self): return self
        @property
        def parent(self): return _FakeParent(tmp_path)
        def __truediv__(self, other): return self._p / other

    (tmp_path / "rules.txt").write_text(
        "Rule 22.15\tMandatory\nNo dynamic memory after init.\n",
        encoding="utf-8",
    )
    with patch.object(ingest_mod, "Path", side_effect=lambda *a: _FakePath(*a)):
        rules = parse_misra_file("rules.txt")

    assert rules[0]["section"] == 22
    assert rules[0]["rule_number"] == 15
    assert rules[0]["category"] == "Mandatory"


def test_parse_misra_scope_is_always_misra_c_2023(tmp_path: Path):
    import app.data.ingest as ingest_mod

    class _FakeParent:
        def __init__(self, root): self._root = root
        @property
        def parent(self): return self
        def __truediv__(self, other): return self._root / other

    class _FakePath:
        def __init__(self, *a): self._p = Path(*a)
        def resolve(self): return self
        @property
        def parent(self): return _FakeParent(tmp_path)
        def __truediv__(self, other): return self._p / other

    (tmp_path / "rules.txt").write_text(
        "Rule 1.1    Required\nScope test.\n", encoding="utf-8"
    )
    with patch.object(ingest_mod, "Path", side_effect=lambda *a: _FakePath(*a)):
        rules = parse_misra_file("rules.txt")

    assert rules[0]["scope"] == "MISRA C:2023"


# ---------------------------------------------------------------------------
# upload_to_mongodb
# ---------------------------------------------------------------------------

async def test_upload_to_mongodb_empty_rules_does_not_call_bulk_write():
    svc = _make_mongodb_service()
    await upload_to_mongodb([], svc)
    svc.collection.bulk_write.assert_not_called()


async def test_upload_to_mongodb_calls_bulk_write_with_replace_one_ops():
    from pymongo import ReplaceOne

    rules = [
        {"rule_type": "RULE", "section": 1, "rule_number": 1, "full_text": "text"},
        {"rule_type": "DIR", "section": 4, "rule_number": 1, "full_text": "text"},
    ]
    svc = _make_mongodb_service()
    await upload_to_mongodb(rules, svc)

    svc.collection.bulk_write.assert_called_once()
    ops = svc.collection.bulk_write.call_args[0][0]
    assert len(ops) == 2
    assert all(isinstance(op, ReplaceOne) for op in ops)


async def test_upload_to_mongodb_create_indexes_error_returns_early():
    """If create_indexes raises, the function logs and returns without calling bulk_write."""
    svc = _make_mongodb_service()
    svc.create_indexes = AsyncMock(side_effect=Exception("index error"))

    rules = [{"rule_type": "RULE", "section": 1, "rule_number": 1, "full_text": "t"}]
    await upload_to_mongodb(rules, svc)

    svc.collection.bulk_write.assert_not_called()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

async def test_main_with_parsed_rules_calls_upload_and_embed():
    """When parse_misra_file returns rules, main calls upload_to_mongodb and embed_and_store."""
    rules = [{"rule_type": "RULE", "section": 1, "rule_number": 1, "full_text": "Rule text"}]

    mock_mongodb = MagicMock()
    mock_pinecone = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_and_store = AsyncMock(return_value=42)

    with patch("app.data.ingest.parse_misra_file", return_value=rules), \
         patch("app.data.ingest.upload_to_mongodb", new=AsyncMock()) as mock_upload:
        result = await main(mock_mongodb, mock_pinecone, mock_embedder)

    mock_upload.assert_called_once_with(rules, mock_mongodb)
    mock_embedder.embed_and_store.assert_called_once_with(rules, mock_pinecone)
    assert result == {"rules_ingested": 1, "vectors_upserted": 42}


async def test_main_returns_zeros_when_no_rules_parsed():
    """When parse_misra_file returns an empty list, main returns zero counts."""
    mock_mongodb = MagicMock()
    mock_pinecone = MagicMock()
    mock_embedder = MagicMock()

    with patch("app.data.ingest.parse_misra_file", return_value=[]):
        result = await main(mock_mongodb, mock_pinecone, mock_embedder)

    assert result == {"rules_ingested": 0, "vectors_upserted": 0}
    mock_embedder.embed_and_store.assert_not_called()


async def test_main_returns_correct_counts_for_multiple_rules():
    rules = [
        {"rule_type": "RULE", "section": i, "rule_number": i, "full_text": f"Rule {i}"}
        for i in range(5)
    ]
    mock_mongodb = MagicMock()
    mock_pinecone = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_and_store = AsyncMock(return_value=5)

    with patch("app.data.ingest.parse_misra_file", return_value=rules), \
         patch("app.data.ingest.upload_to_mongodb", new=AsyncMock()):
        result = await main(mock_mongodb, mock_pinecone, mock_embedder)

    assert result["rules_ingested"] == 5
    assert result["vectors_upserted"] == 5


# ---------------------------------------------------------------------------
# run_ingest_cli()
# ---------------------------------------------------------------------------

async def test_run_ingest_cli_uses_service_container_and_logs_result():
    """run_ingest_cli should enter the service container, call main, and log the result."""
    mock_container = MagicMock()
    mock_container.mongodb = MagicMock()
    mock_container.pinecone = MagicMock()
    mock_container.embedding = MagicMock()

    # Build an async context manager that yields mock_container
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_container():
        yield mock_container

    expected_result = {"rules_ingested": 3, "vectors_upserted": 3}

    with patch("app.data.ingest.create_service_container", return_value=_fake_container()), \
         patch("app.data.ingest.main", new=AsyncMock(return_value=expected_result)) as mock_main:
        await run_ingest_cli()

    mock_main.assert_called_once_with(
        mongodb=mock_container.mongodb,
        pinecone=mock_container.pinecone,
        embedder=mock_container.embedding,
    )

"""End-to-end smoke for the async_chat_cli + the chat --async route.

Verifies:
  * synapforge.demo.async_chat_cli builds an argparse parser cleanly.
  * --transcript replay reaches the listener and produces output.
  * synapforge.demo.cli ``chat --async --transcript`` forwards through
    the same path (the new flags wired into the legacy CLI).

These run on CPU only against the stub StreamingGen; no model checkpoint
is required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pytest


@pytest.fixture
def transcript_file(tmp_path) -> Path:
    p = tmp_path / "smoke.jsonl"
    p.write_text(
        '"hi there"\n"<EOT>"\n"what is CfC"\n"<EOT>"\n"<EOF>"\n',
        encoding="utf-8",
    )
    return p


def test_async_chat_cli_argparser_help():
    from synapforge.demo import async_chat_cli

    with pytest.raises(SystemExit) as exc:
        async_chat_cli.main(["--help"])
    assert exc.value.code == 0


def test_async_chat_cli_transcript_run(tmp_path, transcript_file):
    """End-to-end: feed a transcript, write a session JSON."""
    from synapforge.demo import async_chat_cli

    save = tmp_path / "session.json"
    rc = async_chat_cli.main(
        [
            "--async",
            "--proactive",
            "--transcript",
            str(transcript_file),
            "--save",
            str(save),
            "--target-tps",
            "0",
        ]
    )
    assert rc == 0
    assert save.is_file()
    data = json.loads(save.read_text(encoding="utf-8"))
    assert data["user_messages"] == ["hi there", "what is CfC"]
    assert "policy_stats" in data
    assert "trigger_stats" in data


def test_demo_cli_chat_async_route(tmp_path, transcript_file, monkeypatch):
    """``synapforge-demo chat --async --transcript ...`` reaches async path."""
    from synapforge.demo import cli

    save = tmp_path / "via_demo.json"
    rc = cli.main(
        [
            "chat",
            "--async",
            "--transcript",
            str(transcript_file),
            "--save",
            str(save),
        ]
    )
    assert rc == 0
    assert save.is_file()
    data = json.loads(save.read_text(encoding="utf-8"))
    assert data["user_messages"] == ["hi there", "what is CfC"]


def test_demo_cli_chat_sync_default_unchanged(tmp_path, monkeypatch):
    """Without --async, the chat path is the legacy synchronous demo."""
    from synapforge.demo import cli

    save = tmp_path / "sync.json"
    # Force the recorded-replay (no ckpt).
    rc = cli.main(
        [
            "chat",
            "--ckpt",
            str(tmp_path / "no_such_ckpt.pt"),
            "--save",
            str(save),
        ]
    )
    assert rc == 0
    assert save.is_file()
    data = json.loads(save.read_text(encoding="utf-8"))
    # Synchronous mode produces a 'mode' = 'recorded' or 'live'.
    assert data["mode"] in ("recorded", "live")

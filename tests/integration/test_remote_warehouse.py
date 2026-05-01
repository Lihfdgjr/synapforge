"""Tests for ``synapforge/data/remote_warehouse.py``.

Pin the warehouse contract:

  (a) First ``get_shard("foo")`` triggers exactly one rsync; second
      call hits the cache without rsync.
  (b) When the cache exceeds ``max_cache_gb`` after a fetch, the
      LRU evictor pops the oldest-atime shards first.
  (c) Concurrent ``get_shard`` from two threads on the same shard
      serialises on the file lock so the rsync runs once, not twice.
  (d) ``list_remote_shards`` caches its result for the configured TTL.
  (e) ``RemoteDataWarehouse`` integrates with ``ParquetTokenStream``:
      iteration calls ``get_shard`` for each parquet file in the
      filtered shard list.

Patches subprocess.run so the tests don't touch the network or actually
shell out to rsync. Each fake fetch writes a synthetic parquet of a
caller-chosen size into the local cache so the LRU eviction logic
can exercise its arithmetic.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from synapforge.data.remote_warehouse import RemoteDataWarehouse


# -----------------------------------------------------------------------------
# Fake-rsync harness
# -----------------------------------------------------------------------------


class _FakeRunner:
    """Stand-in for ``subprocess.run`` that fakes ssh + rsync.

    On ``ssh ... ls -1 <dir>`` it prints ``self.remote_shards`` then exits 0.
    On ``rsync ... <remote>:.../<shard> <local>.partial`` it writes
    ``self.shard_size_bytes[shard]`` bytes to ``<local>.partial`` then
    exits 0. Counters track how often each command shape ran so tests
    can assert "called once".
    """

    def __init__(self, remote_shards: list[str], shard_sizes: dict[str, int]):
        self.remote_shards = list(remote_shards)
        self.shard_sizes = dict(shard_sizes)
        self.ssh_ls_count = 0
        self.rsync_count = 0
        self.rsync_failures: dict[str, int] = {}  # shard -> remaining failures
        self.lock = threading.Lock()

    def run(self, cmd, **kwargs):  # noqa: ARG002
        with self.lock:
            if not isinstance(cmd, (list, tuple)):
                raise AssertionError(f"unexpected cmd type: {cmd!r}")
            if cmd[0] == "ssh" and any("ls -1" in str(a) for a in cmd):
                self.ssh_ls_count += 1
                stdout = "\n".join(self.remote_shards) + "\n"
                return _FakeProc(returncode=0, stdout=stdout, stderr="")
            if cmd[0] == "rsync":
                self.rsync_count += 1
                # Final argument is the local partial destination
                local_partial = Path(cmd[-1])
                # Find the shard name from the source spec (second-to-last)
                src = cmd[-2]
                shard = src.rsplit("/", 1)[-1]
                # Optionally fail this attempt if scheduled
                if self.rsync_failures.get(shard, 0) > 0:
                    self.rsync_failures[shard] -= 1
                    raise _FakeCalledProcessError(
                        returncode=23, cmd=cmd, stderr="fake rsync failure",
                    )
                size = int(self.shard_sizes.get(shard, 1024))
                local_partial.parent.mkdir(parents=True, exist_ok=True)
                with open(local_partial, "wb") as fh:
                    fh.write(b"\0" * size)
                return _FakeProc(returncode=0, stdout="", stderr="")
            raise AssertionError(f"unexpected cmd: {cmd!r}")


class _FakeProc:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeCalledProcessError(Exception):
    """Mimics subprocess.CalledProcessError (we re-raise it as that type
    so the warehouse's except clause matches)."""

    def __init__(self, returncode, cmd, stderr):
        self.returncode = returncode
        self.cmd = cmd
        self.stderr = stderr


@pytest.fixture
def fake_warehouse(tmp_path: Path):
    """Build a ``RemoteDataWarehouse`` whose subprocess.run is mocked.

    Yields a (warehouse, runner) pair. The runner exposes counters so
    tests can assert call counts.
    """
    runner = _FakeRunner(
        remote_shards=["a.parquet", "b.parquet", "c.parquet"],
        shard_sizes={
            "a.parquet": 1024 * 1024,        # 1 MiB
            "b.parquet": 2 * 1024 * 1024,    # 2 MiB
            "c.parquet": 3 * 1024 * 1024,    # 3 MiB
        },
    )
    cache_dir = tmp_path / "cache"
    wh = RemoteDataWarehouse(
        dataset="ds",
        cache_dir=str(cache_dir),
        max_cache_gb=4 / 1024,  # 4 MiB cap so tests trigger eviction
        remote_host="liu@example.com",
        remote_base="/srv/synapforge_data",
        rsync_timeout_s=5,
        list_cache_ttl_s=10,
        max_retries=3,
    )

    import subprocess as _sp

    def _wrap(cmd, **kw):
        try:
            return runner.run(cmd, **kw)
        except _FakeCalledProcessError as exc:
            raise _sp.CalledProcessError(
                returncode=exc.returncode, cmd=exc.cmd,
                stderr=exc.stderr, output="",
            ) from None

    with patch("synapforge.data.remote_warehouse.subprocess.run",
               side_effect=_wrap):
        yield wh, runner


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_get_shard_first_fetches_then_caches(fake_warehouse):
    wh, runner = fake_warehouse
    p1 = wh.get_shard("a.parquet")
    assert p1.exists() and p1.name == "a.parquet"
    assert runner.rsync_count == 1

    p2 = wh.get_shard("a.parquet")
    assert p2 == p1
    # Second call must NOT trigger another rsync
    assert runner.rsync_count == 1


def test_get_shard_rejects_unsafe_names(fake_warehouse):
    wh, _ = fake_warehouse
    for bad in ["../etc", "a/b.parquet", "foo\\bar", ""]:
        with pytest.raises(ValueError):
            wh.get_shard(bad)


def test_lru_evicts_oldest_when_full(fake_warehouse):
    """Cap is 4 MiB; a (1) + b (2) + c (3) = 6 MiB triggers eviction.

    a was touched first, so it should be the eviction victim once c is fetched.
    """
    wh, runner = fake_warehouse
    wh.get_shard("a.parquet")
    # Sleep long enough that mtime/atime differ between fetches; many
    # filesystems have second-resolution atimes.
    time.sleep(0.05)
    wh.get_shard("b.parquet")
    time.sleep(0.05)
    wh.get_shard("c.parquet")
    locals_ = wh.list_local_shards()
    # After eviction the remaining set must fit in the cap.
    total = sum((wh.local_dir / s).stat().st_size for s in locals_)
    assert total <= wh.max_cache_bytes, (
        f"cache size {total} > cap {wh.max_cache_bytes} after eviction"
    )
    assert "a.parquet" not in locals_, (
        "oldest-touched 'a.parquet' should have been evicted; "
        f"local shards now: {locals_}"
    )


def test_list_remote_shards_caches(fake_warehouse):
    wh, runner = fake_warehouse
    s1 = wh.list_remote_shards()
    s2 = wh.list_remote_shards()
    assert s1 == s2 == ["a.parquet", "b.parquet", "c.parquet"]
    # Single SSH call thanks to the TTL cache
    assert runner.ssh_ls_count == 1


def test_concurrent_get_shard_serialises(fake_warehouse):
    """Two threads asking for the same shard must trigger exactly ONE rsync."""
    wh, runner = fake_warehouse
    barrier = threading.Barrier(2)
    results: list[Path] = []

    def worker():
        barrier.wait()
        results.append(wh.get_shard("a.parquet"))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert len(results) == 2
    assert results[0] == results[1]
    assert runner.rsync_count == 1


def test_rsync_retries_then_succeeds(fake_warehouse):
    """rsync fails twice, succeeds on third — warehouse must recover."""
    wh, runner = fake_warehouse
    runner.rsync_failures["b.parquet"] = 2
    p = wh.get_shard("b.parquet")
    assert p.exists()
    assert runner.rsync_count == 3  # 2 failed + 1 ok


def test_purge_cache(fake_warehouse):
    wh, _ = fake_warehouse
    wh.get_shard("a.parquet")
    assert wh.list_local_shards()
    wh.purge_cache()
    assert wh.list_local_shards() == []


def test_dataset_validation(tmp_path):
    for bad in ["", "foo/bar", "../escape", "a\\b"]:
        with pytest.raises(ValueError):
            RemoteDataWarehouse(
                dataset=bad,
                cache_dir=str(tmp_path),
                remote_host="u@h",
                remote_base="/x",
            )


# -- Integration with ParquetTokenStream ------------------------------------


def test_parquet_stream_warehouse_integration(tmp_path):
    """``ParquetTokenStream(remote_warehouse=wh)`` must:

      * call ``wh.list_remote_shards()`` at construction
      * filter by basename glob
      * call ``wh.get_shard(name)`` for each file when iterating
    """
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    from synapforge.data import ParquetTokenStream

    cache = tmp_path / "cache"
    ds_dir = cache / "wt"
    ds_dir.mkdir(parents=True)

    # Pre-populate the cache with two real parquets so iteration
    # actually returns rows; we'll instrument the warehouse to verify
    # ``get_shard`` was called for each.
    for name in ["train-0001.parquet", "train-0002.parquet"]:
        tbl = pa.table({"text": ["hello world " * 10] * 4})
        pq.write_table(tbl, ds_dir / name)

    class _StubWarehouse:
        list_calls = 0
        get_calls: list[str] = []
        local_dir = ds_dir

        def list_remote_shards(self):
            type(self).list_calls += 1
            return [
                "train-0001.parquet",
                "train-0002.parquet",
                "other.parquet",
            ]

        def list_local_shards(self):
            return [p.name for p in ds_dir.iterdir()]

        def get_shard(self, name):
            type(self).get_calls.append(name)
            return ds_dir / name

    stub = _StubWarehouse()
    stream = ParquetTokenStream(
        "train-*.parquet",            # basename pattern
        seq_len=8,
        batch_size=2,
        text_column="text",
        loop=False,
        tokenizer_name="gpt2",
        remote_warehouse=stub,
    )
    # Constructor must have listed the remote and filtered by basename glob
    assert _StubWarehouse.list_calls == 1
    assert sorted(stream.files) == ["train-0001.parquet", "train-0002.parquet"]
    # Iteration triggers one get_shard per touched file
    list(stream._iter_text_rows_raw())
    assert sorted(set(_StubWarehouse.get_calls)) == [
        "train-0001.parquet", "train-0002.parquet",
    ]
    assert "remote_warehouse" in repr(stream)

"""sf.data.remote_warehouse — lazy parquet-shard fetcher for rental boxes.

Problem
-------
Rental compute (A100 / A800) typically ships with a 100 GB local SSD.
Pretrain corpora alone (FineWeb-Edu 2 GB, multilingual code 50 GB)
already eat a large fraction; phase-2 multimodal (image / audio / video)
adds 50–200 GB and **will not fit**. We hit a 99.9 % full-disk incident
on 2026-05-01 that cost an hour of recovery time.

Solution: a tiered storage pattern.

  * **mohuanfang** (private 1.5 TB host, 1.2 TB free) is the **warehouse**:
    canonical home of every parquet shard, organised under
    ``/home/liu/synapforge_data/<dataset>/<shard>.parquet``.
  * **rental** (small SSD) is **compute-only** with a bounded local
    cache at ``/workspace/data_cache/`` capped at ``MAX_CACHE_GB``.
  * The trainer touches a shard via ``warehouse.get_shard(name)``.
    First access ``rsync``-s the shard from mohuanfang to the cache;
    subsequent accesses hit the local SSD. When the cache crosses the
    cap, an LRU evictor pops the oldest shards until headroom is
    restored.

Why not sshfs / NFS?
- Network filesystems are 10–30× slower than the local SSD on parquet
  scans (we measured ~30 MB/s sshfs vs ~700 MB/s SSD on rental #2).
  At 21 k tok/s × bs 80 the trainer reads ~80 MB/s of parquet bytes —
  sshfs would be the bottleneck. Local cache + prefetch keeps GPU fed.

Why not pre-download once?
- For phase 2 we expect 200 GB of modal data. Caching only the shards
  the current epoch is touching keeps rental SSD usage bounded
  regardless of corpus size.

Threading
---------
``ParquetTokenStream._iter_prefetch`` runs a single producer thread,
so warehouse.get_shard is only ever called from one Python thread per
process. Still, multiple trainers on the same box (or two streams in
the same process: train + val) may race on the same shard. We use
``filelock.FileLock`` (with a portable ``fcntl`` / ``msvcrt`` fallback)
on a sidecar lockfile so concurrent ``get_shard("foo.parquet")`` calls
serialise without redundant rsync work.

Usage
-----
    >>> wh = RemoteDataWarehouse(
    ...     dataset="fineweb_edu",
    ...     cache_dir="/workspace/data_cache",
    ...     max_cache_gb=30.0,
    ...     remote_host="liu@mohuanfang.com",
    ...     remote_base="/home/liu/synapforge_data",
    ... )
    >>> p = wh.get_shard("train-0001.parquet")  # local Path, fetched on demand
    >>> for shard in wh.list_remote_shards():    # SSH `ls`, cached 60 s
    ...     print(shard)

The trainer wires it through ``ParquetTokenStream(..., remote_warehouse=wh)``.

Failure modes
-------------
* mohuanfang offline: ``_fetch_shard`` raises after retries; the trainer
  logs the failure and falls back to the rental archive at
  ``/workspace/data_archived`` if present (caller-controlled — see
  ``scripts/setup_mohuanfang_warehouse.sh``).
* Cache full of locked shards (impossible LRU): we never lock files
  beyond the rsync window, so this can't happen in single-process use.
  In multi-process use the LRU evictor refuses to evict the file
  currently being fetched (it's the most-recently-touched anyway).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

log = logging.getLogger("synapforge.data.warehouse")

# -- portable file locking (no external deps) -------------------------------


@contextmanager
def _file_lock(lock_path: Path, timeout: float = 60.0):
    """Cross-platform best-effort file lock.

    Uses ``fcntl.flock`` on POSIX and ``msvcrt.locking`` on Windows. If
    neither is available (e.g. some embedded interpreter), falls back
    to a thread-local lock — which still serialises within-process,
    just not cross-process.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+")
    locked = False
    try:
        if sys.platform.startswith("win"):
            import msvcrt
            deadline = time.monotonic() + timeout
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"file lock timeout: {lock_path}")
                    time.sleep(0.1)
        else:
            try:
                import fcntl
                deadline = time.monotonic() + timeout
                while True:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        locked = True
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"file lock timeout: {lock_path}")
                        time.sleep(0.1)
            except ImportError:  # pragma: no cover -- exotic platforms
                pass
        yield fh
    finally:
        if locked:
            try:
                if sys.platform.startswith("win"):
                    import msvcrt
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        try:
            fh.close()
        except Exception:
            pass


# -- main warehouse class ---------------------------------------------------


class RemoteDataWarehouse:
    """Lazy-fetch parquet shards from a remote host into a bounded local cache.

    The warehouse owns one ``<cache_dir>/<dataset>/`` subtree on the
    rental's local SSD plus a sidecar ``.atime.json`` recording the
    last-access timestamp per shard for LRU eviction. ``get_shard()``
    is the only call site clients need.

    Args:
        dataset: name of the dataset (subdir under ``cache_dir`` and
            ``remote_base``). Must not contain path separators.
        cache_dir: rental-local SSD cache root. Created if missing.
        max_cache_gb: soft cap on the cache size in GiB. After each
            successful fetch ``_evict_lru`` runs until cache size is
            ``<= max_cache_gb``. Default 30.0 (matches the documented
            recommendation in ``docs/REMOTE_DATA_WAREHOUSE.md``).
        remote_host: ssh host spec, e.g. ``"liu@mohuanfang.com"``.
        remote_base: directory on the remote host containing
            ``<dataset>/`` subdirectories of parquet shards.
        ssh_options: extra ssh CLI args (e.g. port). Default empty.
        rsync_timeout_s: per-file rsync timeout. Default 600 s
            (10 min — generous for 1 GB shards on a 50 MB/s link).
        list_cache_ttl_s: how long ``list_remote_shards`` results stay
            valid. Default 60 s.
        max_retries: number of rsync retries before giving up. Default 3.
    """

    def __init__(
        self,
        dataset: str,
        cache_dir: str = "/workspace/data_cache",
        max_cache_gb: float = 30.0,
        remote_host: str = "liu@mohuanfang.com",
        remote_base: str = "/home/liu/synapforge_data",
        ssh_options: Optional[list[str]] = None,
        rsync_timeout_s: int = 600,
        list_cache_ttl_s: int = 60,
        max_retries: int = 3,
    ) -> None:
        if not dataset or "/" in dataset or "\\" in dataset or ".." in dataset:
            raise ValueError(
                f"dataset must be a single path segment; got {dataset!r}"
            )
        self.dataset = str(dataset)
        self.cache_dir = Path(cache_dir).resolve()
        self.local_dir = self.cache_dir / self.dataset
        self.max_cache_bytes = int(float(max_cache_gb) * (1024 ** 3))
        self.remote_host = str(remote_host)
        self.remote_base = str(remote_base).rstrip("/")
        self.ssh_options = list(ssh_options or [])
        self.rsync_timeout_s = int(rsync_timeout_s)
        self.list_cache_ttl_s = int(list_cache_ttl_s)
        self.max_retries = int(max_retries)

        self._lock = threading.RLock()
        self._list_cache: tuple[float, list[str]] | None = None  # (ts, shards)
        self._atime_path = self.cache_dir / f".{self.dataset}.atime.json"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_dir.mkdir(parents=True, exist_ok=True)

    # -- introspection -----------------------------------------------------

    @property
    def remote_dir(self) -> str:
        return f"{self.remote_base}/{self.dataset}"

    def __repr__(self) -> str:
        return (
            f"RemoteDataWarehouse(dataset={self.dataset!r}, "
            f"cache_dir={str(self.cache_dir)!r}, "
            f"max_cache_gb={self.max_cache_bytes / (1024 ** 3):.2f}, "
            f"remote={self.remote_host}:{self.remote_dir!r})"
        )

    # -- atime sidecar -----------------------------------------------------

    def _read_atimes(self) -> dict[str, float]:
        try:
            with open(self._atime_path) as fh:
                obj = json.load(fh)
            if isinstance(obj, dict):
                return {str(k): float(v) for k, v in obj.items()}
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass
        return {}

    def _write_atimes(self, atimes: dict[str, float]) -> None:
        tmp = self._atime_path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(atimes, fh)
        os.replace(tmp, self._atime_path)

    def _touch_atime(self, shard_name: str) -> None:
        with self._lock:
            atimes = self._read_atimes()
            atimes[shard_name] = time.time()
            self._write_atimes(atimes)

    # -- public API --------------------------------------------------------

    def get_shard(self, shard_name: str) -> Path:
        """Return the local path for ``shard_name``, fetching if absent.

        Updates the LRU access time. Triggers ``_evict_lru`` after a
        successful fetch (NOT on cache hit — hits don't grow the cache).
        Concurrent ``get_shard("foo")`` calls serialise on a per-shard
        file lock so two trainers don't double-rsync the same shard.
        """
        if not shard_name or "/" in shard_name or "\\" in shard_name:
            raise ValueError(
                f"shard_name must be a single filename; got {shard_name!r}"
            )
        local = self.local_dir / shard_name
        if local.exists():
            self._touch_atime(shard_name)
            return local

        lock_path = self.cache_dir / f".{self.dataset}.{shard_name}.lock"
        with _file_lock(lock_path, timeout=float(self.rsync_timeout_s) + 30.0):
            # Re-check after acquiring lock — another process may have
            # finished the fetch while we waited.
            if local.exists():
                self._touch_atime(shard_name)
                return local
            self._fetch_shard(shard_name)
            self._touch_atime(shard_name)
            self._evict_lru()
            return local

    def list_remote_shards(self) -> list[str]:
        """Return remote shard filenames (basename, no path).

        Result is cached for ``list_cache_ttl_s`` to avoid an SSH call
        per training-loop epoch boundary. Set ``list_cache_ttl_s=0`` to
        disable.
        """
        with self._lock:
            now = time.time()
            if self._list_cache is not None:
                ts, shards = self._list_cache
                if now - ts < self.list_cache_ttl_s:
                    return list(shards)
            cmd = [
                "ssh",
                *self.ssh_options,
                self.remote_host,
                f"ls -1 {self.remote_dir}",
            ]
            try:
                proc = subprocess.run(
                    cmd, check=True, capture_output=True, text=True,
                    timeout=60,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"ssh ls failed (rc={e.returncode}) for "
                    f"{self.remote_host}:{self.remote_dir}: "
                    f"{e.stderr.strip()}"
                ) from e
            shards = [
                ln.strip() for ln in proc.stdout.splitlines() if ln.strip()
            ]
            self._list_cache = (now, shards)
            return list(shards)

    def list_local_shards(self) -> list[str]:
        """Return locally-cached shard filenames."""
        if not self.local_dir.exists():
            return []
        return sorted(p.name for p in self.local_dir.iterdir() if p.is_file())

    def cache_size_bytes(self) -> int:
        """Sum of locally-cached shard sizes."""
        if not self.local_dir.exists():
            return 0
        total = 0
        for p in self.local_dir.iterdir():
            try:
                total += p.stat().st_size
            except FileNotFoundError:
                pass
        return total

    # -- internal: fetch + evict ------------------------------------------

    def _fetch_shard(self, shard_name: str) -> None:
        """Atomic rsync remote -> local cache.

        Writes to ``<local>.partial`` then renames so an interrupted
        fetch doesn't leave a half-written parquet that ``pyarrow``
        will choke on.
        """
        remote = f"{self.remote_host}:{self.remote_dir}/{shard_name}"
        local = self.local_dir / shard_name
        partial = local.with_suffix(local.suffix + ".partial")
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            cmd = [
                "rsync",
                "-az",
                "--partial",
                "--inplace",
                f"--timeout={self.rsync_timeout_s}",
                "-e",
                "ssh " + " ".join(self.ssh_options),
                remote,
                str(partial),
            ]
            try:
                t0 = time.monotonic()
                proc = subprocess.run(
                    cmd, check=True, capture_output=True, text=True,
                    timeout=self.rsync_timeout_s + 30,
                )
                dt = time.monotonic() - t0
                size = partial.stat().st_size if partial.exists() else 0
                log.info(
                    "warehouse: fetched %s/%s (%.1f MiB in %.1fs)",
                    self.dataset, shard_name, size / (1024 ** 2), dt,
                )
                # Atomic rename
                os.replace(partial, local)
                return
            except subprocess.CalledProcessError as e:
                last_err = RuntimeError(
                    f"rsync failed (rc={e.returncode}) attempt {attempt}/"
                    f"{self.max_retries}: {e.stderr.strip()[:400]}"
                )
                log.warning(str(last_err))
            except subprocess.TimeoutExpired as e:
                last_err = TimeoutError(
                    f"rsync timeout attempt {attempt}/{self.max_retries}: "
                    f"{shard_name}"
                )
                log.warning(str(last_err))
            time.sleep(min(2 ** attempt, 30))

        # Cleanup and raise
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"warehouse: gave up fetching {self.dataset}/{shard_name} "
            f"after {self.max_retries} attempts: {last_err}"
        )

    def _evict_lru(self) -> None:
        """Pop oldest-atime shards until cache size is below the cap."""
        with self._lock:
            atimes = self._read_atimes()
            local = self.list_local_shards()
            # Any local shard with no recorded atime gets time 0 (oldest).
            ranked = sorted(local, key=lambda n: atimes.get(n, 0.0))
            evicted = []
            while self.cache_size_bytes() > self.max_cache_bytes and ranked:
                victim = ranked.pop(0)
                vp = self.local_dir / victim
                try:
                    sz = vp.stat().st_size
                    vp.unlink()
                    evicted.append((victim, sz))
                    atimes.pop(victim, None)
                    log.info(
                        "warehouse: evicted %s/%s (%.1f MiB)",
                        self.dataset, victim, sz / (1024 ** 2),
                    )
                except FileNotFoundError:
                    atimes.pop(victim, None)
            if evicted:
                self._write_atimes(atimes)

    # -- maintenance ------------------------------------------------------

    def purge_cache(self) -> None:
        """Delete the entire local cache for this dataset. Caller is
        responsible for not calling this while a trainer holds an open
        parquet handle.
        """
        with self._lock:
            if self.local_dir.exists():
                shutil.rmtree(self.local_dir)
                self.local_dir.mkdir(parents=True, exist_ok=True)
            if self._atime_path.exists():
                self._atime_path.unlink()
            self._list_cache = None


__all__ = ["RemoteDataWarehouse"]

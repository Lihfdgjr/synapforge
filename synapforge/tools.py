"""sf.tools — first-class tool-use for synapforge.

Eight components, ported and adapted from mscfc/tools.py. Crucial difference
from the mscfc port: the ``ToolCaller`` in *this* module is a ``sf.Module``
that does **direct neural dispatch** (hidden state -> tool selection
distribution) — there is NO ``<tool_call>`` JSON token, no MCP wire protocol.
This mirrors ``synapforge.action.NeuroMCPHead``: the model picks tools the
same way it picks actions — via a learned codebook of tool prototypes.

External services (web search / fetch / scrape) are thin wrappers that
gracefully degrade to mock results when the network is unreachable, so
smoke tests stay deterministic under ``--no-network``.

Components
----------
    ToolRegistry       declarative ``{name -> (callable, schema)}`` manifest
    WebSearchTool      DDG-default; falls back to BingHTML; ``mock=True`` returns canned results
    WebFetchTool       urllib + BeautifulSoup main-text extraction
    WebScrapeTool      bs4 + CSS selectors
    ShellTool          subprocess with strict whitelist
    CodeExecTool       firejail/docker sandbox; refuses bare-exec
    ToolCaller         sf.Module — neural dispatch, NOT token-protocol
    ToolLearningLoop   Toolformer-style self-supervised tool reward
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import Module

# ---------------------------------------------------------------------------
# 1. ToolRegistry
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., Any]
    schema: dict[str, Any]
    description: str = ""


class ToolRegistry:
    """Declarative ``{name -> ToolSpec}`` manifest.

    Order of registration is preserved — ``ToolCaller`` indexes tools by
    insertion order so the neural dispatch maps `output_dict[name]` 1:1
    with prototype slots.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        schema: dict[str, Any] | None = None,
        description: str = "",
    ) -> ToolSpec:
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if not callable(fn):
            raise TypeError("fn must be callable")
        spec = ToolSpec(name=name, fn=fn, schema=dict(schema or {}), description=description)
        self._tools[name] = spec
        return spec

    def add_default_tools(
        self,
        web_search_mock: bool = True,
        web_fetch_mock: bool = True,
        shell: bool = False,
        code_exec: bool = False,
    ) -> ToolRegistry:
        """Convenience: register a sane default set.

        Defaults to mocked web tools so smoke tests stay deterministic.
        Real services are opted in via ``web_search_mock=False`` etc.
        """
        ws = WebSearchTool(mock=web_search_mock)
        self.register("web_search", ws.call, schema={"query": "str"}, description="Web search")
        wf = WebFetchTool(mock=web_fetch_mock)
        self.register("web_fetch", wf.call, schema={"url": "str"}, description="Fetch URL text")
        wsr = WebScrapeTool()
        self.register(
            "web_scrape",
            wsr.call,
            schema={"url": "str", "selectors": "dict"},
            description="CSS-selector scrape",
        )
        if shell:
            sh = ShellTool()
            self.register("shell", sh.call, schema={"cmd": "str"}, description="Whitelisted shell")
        if code_exec:
            try:
                ce = CodeExecTool()
                self.register(
                    "code_exec", ce.call, schema={"code": "str"}, description="Sandboxed Python"
                )
            except RuntimeError:
                # No sandbox installed; silently skip.
                pass
        return self

    def list(self) -> list[dict[str, Any]]:
        return [
            {"name": s.name, "schema": s.schema, "description": s.description}
            for s in self._tools.values()
        ]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def has(self, name: str) -> bool:
        return name in self._tools

    def call(self, name: str, args: dict[str, Any] | None = None) -> Any:
        if name not in self._tools:
            raise KeyError(f"tool '{name}' not registered")
        spec = self._tools[name]
        return spec.fn(**(args or {}))

    def __len__(self) -> int:
        return len(self._tools)


# ---------------------------------------------------------------------------
# 2. WebSearchTool
# ---------------------------------------------------------------------------


class _RateLimiter:
    def __init__(self, per_min: int = 10) -> None:
        self.per_min = int(per_min)
        self._events: list[float] = []

    def check(self) -> None:
        now = time.time()
        self._events = [t for t in self._events if now - t < 60.0]
        if len(self._events) >= self.per_min:
            raise RuntimeError(f"rate limit exceeded ({self.per_min}/min)")
        self._events.append(now)


_MOCK_SEARCH_RESULTS = [
    {"title": "Mock result A", "snippet": "synapforge mock snippet A", "url": "https://example.com/a"},
    {"title": "Mock result B", "snippet": "synapforge mock snippet B", "url": "https://example.com/b"},
    {"title": "Mock result C", "snippet": "synapforge mock snippet C", "url": "https://example.com/c"},
]


# ---------------------------------------------------------------------------
# SSRF guard: reject URLs targeting localhost / loopback / link-local / cloud
# metadata. Applied to every model/agent-controlled URL fetched by
# WebFetchTool, WebScrapeTool and the WebSearchTool internal _http_get.
# ---------------------------------------------------------------------------

_BLOCKED_HOST_LITERALS = {
    "localhost",
    "0.0.0.0",
    "::",
    "::1",
    "metadata.google.internal",
    "metadata",
    "169.254.169.254",
}


def _is_blocked_url(url: str) -> tuple[bool, str]:
    """Return (blocked, reason). True if URL targets a forbidden host/scheme.

    Blocks:
      - non-http(s) schemes (file://, gopher://, ftp://, data:, etc.)
      - loopback IPv4/IPv6 (127.0.0.0/8, ::1)
      - link-local (169.254.0.0/16, fe80::/10)
      - private RFC1918 (10/8, 172.16/12, 192.168/16)
      - localhost / known cloud-metadata endpoints
    """
    try:
        from ipaddress import ip_address
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        return True, f"unparsable url: {exc}"
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return True, f"scheme not allowed: {scheme!r}"
    host = (parsed.hostname or "").lower()
    if not host:
        return True, "missing host"
    if host in _BLOCKED_HOST_LITERALS:
        return True, f"blocked host literal: {host!r}"
    # IP-address shortcut
    try:
        ip = ip_address(host)
    except ValueError:
        return False, ""
    if ip.is_loopback or ip.is_link_local or ip.is_private or ip.is_multicast \
            or ip.is_reserved or ip.is_unspecified:
        return True, f"blocked ip class: {host}"
    return False, ""


class WebSearchTool:
    """Web search; engines: duckduckgo (default), bing, baidu, mock.

    ``mock=True`` returns a deterministic canned result so unit tests don't
    need network access. Real engines fall back to mock when unreachable.
    """

    ENGINES = {"duckduckgo", "bing", "baidu", "mock"}

    def __init__(
        self,
        engine: str = "duckduckgo",
        top_k: int = 5,
        timeout_s: int = 10,
        mock: bool = False,
    ) -> None:
        eng = engine.lower()
        if eng not in self.ENGINES:
            raise ValueError(f"engine must be one of {sorted(self.ENGINES)}")
        if mock:
            eng = "mock"
        self.engine = eng
        self.top_k = int(top_k)
        self.timeout_s = int(timeout_s)
        self._limiter = _RateLimiter(per_min=10)

    def _http_get(self, url: str, headers: dict[str, str] | None = None) -> bytes:
        blocked, reason = _is_blocked_url(url)
        if blocked:
            raise PermissionError(f"SSRF guard blocked url: {reason}")
        req = urllib.request.Request(
            url, headers=headers or {"User-Agent": "synapforge-tools/1.0"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return resp.read()

    def _ddg(self, query: str) -> list[dict[str, str]]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return list(_MOCK_SEARCH_RESULTS[: self.top_k])
        q = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={q}"
        try:
            html = self._http_get(url, headers={"User-Agent": "Mozilla/5.0 synapforge"})
        except (urllib.error.URLError, OSError):
            return self._bing_html(query)
        soup = BeautifulSoup(html, "html.parser")
        out: list[dict[str, str]] = []
        for res in soup.select("div.result, div.web-result")[: self.top_k * 3]:
            a = res.select_one("a.result__a, a.result__snippet")
            snip = res.select_one(".result__snippet, .result__body")
            if not a:
                continue
            href = a.get("href", "")
            if "uddg=" in href:
                try:
                    href = urllib.parse.unquote(href.split("uddg=", 1)[1].split("&", 1)[0])
                except Exception:
                    pass
            out.append(
                {
                    "title": a.get_text(strip=True),
                    "snippet": snip.get_text(strip=True) if snip else "",
                    "url": href,
                }
            )
            if len(out) >= self.top_k:
                break
        if not out:
            return self._bing_html(query)
        return out

    def _bing_html(self, query: str) -> list[dict[str, str]]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return list(_MOCK_SEARCH_RESULTS[: self.top_k])
        q = urllib.parse.quote_plus(query)
        url = f"https://www.bing.com/search?q={q}"
        try:
            html = self._http_get(url, headers={"User-Agent": "Mozilla/5.0 synapforge"})
        except (urllib.error.URLError, OSError):
            return list(_MOCK_SEARCH_RESULTS[: self.top_k])
        soup = BeautifulSoup(html, "html.parser")
        out: list[dict[str, str]] = []
        for res in soup.select("li.b_algo")[: self.top_k * 3]:
            a = res.select_one("h2 a") or res.select_one("a")
            snip = res.select_one("p, .b_caption p")
            if not a:
                continue
            out.append(
                {
                    "title": a.get_text(strip=True),
                    "snippet": snip.get_text(strip=True) if snip else "",
                    "url": a.get("href", ""),
                }
            )
            if len(out) >= self.top_k:
                break
        return out

    def _bing(self, query: str) -> list[dict[str, str]]:
        key = os.environ.get("BING_API_KEY")
        if not key:
            return self._bing_html(query)
        q = urllib.parse.quote_plus(query)
        url = f"https://api.bing.microsoft.com/v7.0/search?q={q}&count={self.top_k}"
        try:
            data = self._http_get(url, headers={"Ocp-Apim-Subscription-Key": key})
            obj = json.loads(data)
        except Exception:
            return self._bing_html(query)
        return [
            {
                "title": r.get("name", ""),
                "snippet": r.get("snippet", ""),
                "url": r.get("url", ""),
            }
            for r in obj.get("webPages", {}).get("value", [])[: self.top_k]
        ]

    def call(self, query: str) -> list[dict[str, str]]:
        if self.engine == "mock":
            return list(_MOCK_SEARCH_RESULTS[: self.top_k])
        try:
            self._limiter.check()
        except RuntimeError:
            return list(_MOCK_SEARCH_RESULTS[: self.top_k])
        if self.engine == "duckduckgo":
            return self._ddg(query)
        if self.engine == "bing":
            return self._bing(query)
        # baidu
        return list(_MOCK_SEARCH_RESULTS[: self.top_k])


# ---------------------------------------------------------------------------
# 3. WebFetchTool
# ---------------------------------------------------------------------------


class WebFetchTool:
    """Fetch a URL and return main text + title.

    ``mock=True`` returns canned content. Network failures degrade to a
    short error stub so a calling pipeline never crashes the whole train
    loop on a 503.
    """

    def __init__(
        self,
        timeout_s: int = 10,
        max_bytes: int = 2_000_000,
        mock: bool = False,
    ) -> None:
        self.timeout_s = int(timeout_s)
        self.max_bytes = int(max_bytes)
        self.mock = bool(mock)

    def call(self, url: str) -> dict[str, Any]:
        if self.mock:
            return {
                "url": url,
                "title": "Mock title",
                "text": "synapforge mock fetched body for " + url,
                "length": 40,
                "truncated": False,
            }
        blocked, reason = _is_blocked_url(url)
        if blocked:
            return {"url": url, "title": "", "text": "", "length": 0,
                    "error": f"SSRF guard blocked url: {reason}"}
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "synapforge-tools/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read(self.max_bytes + 1)
        except (urllib.error.URLError, OSError) as exc:
            return {"url": url, "title": "", "text": "", "length": 0, "error": str(exc)}
        truncated = len(raw) > self.max_bytes
        raw = raw[: self.max_bytes]
        text = ""
        title = ""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(raw, "html.parser")
            text = soup.get_text(" ", strip=True)
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
        except ImportError:
            text = raw.decode("utf-8", errors="ignore")
        return {
            "url": url,
            "title": title,
            "text": text,
            "length": len(text),
            "truncated": truncated,
        }


# ---------------------------------------------------------------------------
# 4. WebScrapeTool
# ---------------------------------------------------------------------------


class WebScrapeTool:
    """Structured scrape via BeautifulSoup CSS selectors. Optional Playwright."""

    def __init__(self, timeout_s: int = 10, use_playwright: bool = False) -> None:
        self.timeout_s = int(timeout_s)
        self.use_playwright = bool(use_playwright)

    def _fetch_html(self, url: str) -> bytes:
        blocked, reason = _is_blocked_url(url)
        if blocked:
            raise PermissionError(f"SSRF guard blocked url: {reason}")
        req = urllib.request.Request(url, headers={"User-Agent": "synapforge-tools/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return resp.read()

    def call(self, url: str, selectors: dict[str, str]) -> dict[str, Any]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {key: [] for key in selectors}
        try:
            html = self._fetch_html(url)
        except (urllib.error.URLError, OSError):
            return {key: [] for key in selectors}
        soup = BeautifulSoup(html, "html.parser")
        out: dict[str, Any] = {}
        for key, sel in selectors.items():
            try:
                nodes = soup.select(sel)
            except Exception:
                nodes = []
            out[key] = [n.get_text(" ", strip=True) for n in nodes]
        return out


# ---------------------------------------------------------------------------
# 5. ShellTool
# ---------------------------------------------------------------------------


class ShellTool:
    """Strictly whitelisted subprocess. ``shell=False``; metachars rejected."""

    FORBIDDEN_CHARS = {";", "|", "&", ">", "<", "`", "$("}

    def __init__(
        self,
        whitelist: list[str] | None = None,
        timeout_s: int = 10,
        max_output: int = 64 * 1024,
    ) -> None:
        self.whitelist = list(
            whitelist or ["ls", "cat", "head", "tail", "grep", "find", "wc", "echo", "pwd"]
        )
        self.timeout_s = int(timeout_s)
        self.max_output = int(max_output)

    def call(self, cmd: str) -> dict[str, Any]:
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("cmd must be a non-empty string")
        for bad in self.FORBIDDEN_CHARS:
            if bad in cmd:
                raise PermissionError(f"forbidden metacharacter in cmd: {bad!r}")
        parts = shlex.split(cmd)
        if not parts:
            raise ValueError("empty argv")
        program = os.path.basename(parts[0])
        if program not in self.whitelist:
            raise PermissionError(f"program '{program}' not whitelisted")
        t0 = time.time()
        try:
            proc = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": (exc.stdout or "")[: self.max_output] if exc.stdout else "",
                "stderr": "timeout",
                "exit_code": -1,
                "duration_ms": int((time.time() - t0) * 1000),
            }
        return {
            "stdout": proc.stdout[: self.max_output],
            "stderr": proc.stderr[: self.max_output],
            "exit_code": proc.returncode,
            "duration_ms": int((time.time() - t0) * 1000),
        }


# ---------------------------------------------------------------------------
# 6. CodeExecTool
# ---------------------------------------------------------------------------


class CodeExecTool:
    """Sandboxed Python (firejail / docker). Refuses bare subprocess."""

    def __init__(
        self,
        sandbox: str = "firejail",
        timeout_s: int = 10,
        python_bin: str = "python3",
        docker_image: str = "python:3.11-slim",
        max_output: int = 64 * 1024,
    ) -> None:
        if sandbox not in {"firejail", "docker", "none"}:
            raise ValueError("sandbox must be firejail|docker|none")
        if sandbox == "none":
            raise RuntimeError(
                "CodeExecTool refuses to run without a sandbox. "
                "Install firejail or docker, then pass sandbox='firejail' or 'docker'."
            )
        if sandbox == "firejail" and not shutil.which("firejail"):
            raise RuntimeError("firejail not found on PATH")
        if sandbox == "docker" and not shutil.which("docker"):
            raise RuntimeError("docker not found on PATH")
        self.sandbox = sandbox
        self.timeout_s = int(timeout_s)
        self.python_bin = str(python_bin)
        self.docker_image = str(docker_image)
        self.max_output = int(max_output)

    def _argv(self, code: str) -> list[str]:
        if self.sandbox == "firejail":
            return [
                "firejail",
                "--quiet",
                "--net=none",
                "--private-tmp",
                "--",
                self.python_bin,
                "-c",
                code,
            ]
        return [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--memory=256m",
            "--cpus=1",
            self.docker_image,
            self.python_bin,
            "-c",
            code,
        ]

    def call(self, code: str) -> dict[str, Any]:
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        argv = self._argv(code)
        t0 = time.time()
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": (exc.stdout or "")[: self.max_output] if exc.stdout else "",
                "stderr": "timeout",
                "exit_code": -1,
                "duration_ms": int((time.time() - t0) * 1000),
            }
        return {
            "stdout": proc.stdout[: self.max_output],
            "stderr": proc.stderr[: self.max_output],
            "exit_code": proc.returncode,
            "duration_ms": int((time.time() - t0) * 1000),
        }


# ---------------------------------------------------------------------------
# 7. ToolCaller — neural dispatch (NO TOKEN)
# ---------------------------------------------------------------------------


class ToolCaller(Module):
    """Direct neural tool dispatch — NeuroMCP-style, no JSON / no MCP / no tokens.

    Hidden state ``[*, hidden]`` is mapped to a softmax over
    ``len(registry)`` tool prototypes. Calling forward returns a dict
    ``{tool_name: scalar_or_tensor}`` where the value is the *gated output*
    of executing that tool — when ``execute=True`` we run the tool and
    encode its result back into a tensor; when ``execute=False`` we only
    return the dispatch logits + soft-selection mask.

    The math:
        proto: [N, hidden]    -- learnable tool prototypes
        logits = hidden @ proto.T / sqrt(hidden)        # [*, N]
        sel = softmax(logits) * gate(hidden)            # [*, N]
        # selection is differentiable; argmax is taken for actual exec.

    This is the same primitive used by ``synapforge.action.NeuroMCPHead`` —
    no tool token is ever emitted, and there's no parser to break.
    """

    def __init__(
        self,
        hidden: int,
        registry: ToolRegistry,
        gate: bool = True,
        temperature: float = 1.0,
        execute: bool = False,
    ) -> None:
        super().__init__()
        if len(registry) == 0:
            raise ValueError("registry is empty; add tools first")
        self.hidden = int(hidden)
        self.registry = registry
        self.tool_names = list(registry.names())
        self.n_tools = len(self.tool_names)
        # Tool prototype codebook -- one row per tool.
        self.proto = nn.Parameter(torch.randn(self.n_tools, self.hidden) * (self.hidden ** -0.5))
        # Soft gate: hidden -> [0, 1] dispatch confidence.
        self.gate_proj = nn.Linear(self.hidden, 1) if gate else None
        self.temperature = float(temperature)
        self.execute = bool(execute)
        # Pre-call result cache (token-free; tensors only).
        self._result_dim = self.hidden
        self.result_proj = nn.Linear(self.hidden, self.hidden, bias=False)

    def dispatch(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, selection) where selection is softmax * gate.

        logits:    [*, n_tools]
        selection: [*, n_tools]
        """
        # logits = hidden @ proto.T / sqrt(hidden)
        scale = self.hidden ** -0.5
        proto = self.proto.to(hidden.dtype)
        logits = torch.einsum("...d,nd->...n", hidden, proto) * scale
        logits = logits / max(self.temperature, 1e-6)
        sel = F.softmax(logits, dim=-1)
        if self.gate_proj is not None:
            g = torch.sigmoid(self.gate_proj(hidden))
            sel = sel * g
        return logits, sel

    def _encode_result(
        self, raw: Any, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Turn a tool's raw return into a fixed-size tensor for re-injection.

        Convention: encode the result's string representation as a
        deterministic byte-hash projected through ``result_proj``. Pure
        token-free; nothing about this is parseable as JSON to the model.
        """
        try:
            text = json.dumps(raw, default=str, ensure_ascii=False)
        except Exception:
            text = str(raw)
        # Deterministic hash -> dense tensor.
        b = text.encode("utf-8", errors="ignore")
        if not b:
            buf = torch.zeros(self.hidden, dtype=dtype, device=device)
        else:
            # Cycle bytes into hidden positions; mod-256 normalised to [-1, 1).
            arr = torch.tensor(
                [(bb / 128.0) - 1.0 for bb in b[: self.hidden]],
                dtype=dtype,
                device=device,
            )
            if arr.numel() < self.hidden:
                arr = F.pad(arr, (0, self.hidden - arr.numel()))
            buf = arr
        return self.result_proj(buf.unsqueeze(0)).squeeze(0)

    def forward(
        self,
        hidden: torch.Tensor,
        args_by_tool: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Returns ``{tool_name: output_tensor}``.

        ``output_tensor`` is the gated selection scalar by default. If
        ``self.execute`` is True we additionally execute the *argmax* tool
        and project its result into ``hidden``-dim space; the returned
        dict adds ``"_executed": name`` and ``"_result_vec": tensor``.
        """
        if hidden.dim() < 1 or hidden.shape[-1] != self.hidden:
            raise ValueError(
                f"hidden last-dim mismatch: got {hidden.shape[-1]}, expected {self.hidden}"
            )
        logits, sel = self.dispatch(hidden)
        # Detached selection scalars per tool, indexed by name.
        out: dict[str, Any] = {}
        for i, name in enumerate(self.tool_names):
            out[name] = sel[..., i]
        out["_logits"] = logits
        out["_selection"] = sel
        if self.execute:
            # argmax over the last reduction dim — average sel over leading dims first.
            flat_sel = sel.reshape(-1, self.n_tools).mean(dim=0)
            top = int(flat_sel.argmax().item())
            top_name = self.tool_names[top]
            args = (args_by_tool or {}).get(top_name, {})
            try:
                result = self.registry.call(top_name, args)
            except Exception as exc:  # noqa: BLE001
                result = {"error": str(exc)}
            res_vec = self._encode_result(result, hidden.dtype, hidden.device)
            out["_executed"] = top_name
            out["_result"] = result
            out["_result_vec"] = res_vec
        return out

    def add_tool(self, fresh_proto: torch.Tensor | None = None) -> None:
        """Grow the prototype codebook by one slot (called when registry grows)."""
        names = list(self.registry.names())
        new = len(names) - self.n_tools
        if new <= 0:
            return
        with torch.no_grad():
            extra = (
                fresh_proto.unsqueeze(0)
                if fresh_proto is not None and fresh_proto.dim() == 1
                else torch.randn(new, self.hidden) * (self.hidden ** -0.5)
            )
            new_proto = torch.cat([self.proto.data, extra.to(self.proto.dtype)], dim=0)
        self.proto = nn.Parameter(new_proto)
        self.tool_names = names
        self.n_tools = len(names)


# ---------------------------------------------------------------------------
# 8. ToolLearningLoop — Toolformer-style self-supervised reward
# ---------------------------------------------------------------------------


class _DummyNeuromodulator:
    def __init__(self) -> None:
        self.history: list[tuple[str, float]] = []

    def reinforce(self, tag: str, signal: float) -> None:
        self.history.append((str(tag), float(signal)))


class ToolLearningLoop:
    """Reinforce useful tools, demote useless ones.

    Caller provides a *post-call NLL reduction* (positive = the tool's
    output reduced the model's loss vs no tool). We send a signed signal
    to the neuromodulator and stash failures in ``adversarial_buf`` for
    the self-play loop to revisit.

    No tokens involved — the reward is on the dispatch *selection scalar*
    of the chosen tool, so gradients flow through ``ToolCaller.proto``.
    """

    def __init__(
        self,
        caller: ToolCaller,
        neuromodulator: Any | None = None,
        success_thresh: float = 0.0,
    ) -> None:
        self.caller = caller
        self.neuromodulator = neuromodulator or _DummyNeuromodulator()
        self.success_thresh = float(success_thresh)
        self.adversarial_buf: list[dict[str, Any]] = []

    def evaluate(
        self,
        tool_name: str,
        post_call_nll_reduction: float,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        signal = float(post_call_nll_reduction)
        tag = f"tool:{tool_name}"
        self.neuromodulator.reinforce(tag, signal)
        if signal > self.success_thresh:
            return {"name": tool_name, "signal": signal, "outcome": "success"}
        self.adversarial_buf.append(
            {
                "name": tool_name,
                "args": dict(args or {}),
                "signal": signal,
            }
        )
        return {"name": tool_name, "signal": signal, "outcome": "failure"}

    def reinforce_loss(
        self, hidden: torch.Tensor, tool_name: str, signal: float
    ) -> torch.Tensor:
        """Differentiable surrogate: -signal * log(sel[tool_name]).

        Caller can add this to its outer loss. ``signal>0`` pushes the
        prototype toward ``hidden``; ``signal<0`` pushes it away.
        """
        if tool_name not in self.caller.tool_names:
            raise KeyError(tool_name)
        idx = self.caller.tool_names.index(tool_name)
        _logits, sel = self.caller.dispatch(hidden)
        sel_t = sel[..., idx].clamp(min=1e-8)
        return -float(signal) * sel_t.log().mean()


__all__ = [
    "ToolSpec",
    "ToolRegistry",
    "WebSearchTool",
    "WebFetchTool",
    "WebScrapeTool",
    "ShellTool",
    "CodeExecTool",
    "ToolCaller",
    "ToolLearningLoop",
]

#!/usr/bin/env python3
"""qwen-ui — local web UI for chat + continuous agents.

Runs a small stdlib HTTP server on 127.0.0.1:8001. The chat path is an
SSE proxy in front of qwen-proxy:8000 with a tool-call loop running
locally (so the browser never executes tools). The agents path manages
persisted per-user agent configs in ~/.qwen/agents/<id>/ and spawns
qwen_agent_runner.py subprocesses that survive UI restarts.

Stdlib-only — same constraint as qwen_proxy.py and chat.py.

Usage:
    python qwen_ui.py [--host 127.0.0.1] [--port 8001]

Environment:
    QWEN_HOST, QWEN_PORT — upstream qwen-proxy (default 127.0.0.1:8000)
    QWEN_MODEL_NAME      — model id passed to upstream
    QWEN_UI_HOME         — override ~/.qwen state dir (default $HOME/.qwen)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import heapq
import json
import logging
import mimetypes
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Reuse the agent's tool registry so the chat tab and the agents both
# dispatch through identical code paths. agent_tools.py is on sys.path
# once we add the scripts directory.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)


def _ensure_scripts_on_path() -> None:
    """Idempotent — historic call sites in this file did
    `sys.path.insert(0, scripts_dir)` on every handler invocation, growing
    sys.path unboundedly. This helper inserts only if the path isn't there
    already, so repeated calls don't pollute."""
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)


# Imported lazily inside handlers to keep startup fast (agent_tools pulls in
# urllib3 / various tool helpers, ~80 ms cold). qwen_proxy is light.
from qwen_proxy import parse_tool_calls, split_reasoning  # noqa: E402


# --------------------------------------------------------------------------
# config / paths
# --------------------------------------------------------------------------

UPSTREAM_HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
if UPSTREAM_HOST in ("0.0.0.0", ""):
    UPSTREAM_HOST = "127.0.0.1"
UPSTREAM_PORT = int(os.environ.get("QWEN_PORT", "8000"))
UPSTREAM_BASE = f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}"
# `qwen3.6` is the human-readable alias from the env. dflash-serve refuses
# anything that isn't the exact loaded id (e.g.
# `./models/Qwen3.6-35B-A3B-OptiQ-4bit`). We resolve the live id from
# /v1/models the first time we need it; the env value is used as a label
# in the UI when present.
MODEL_ALIAS = os.environ.get("QWEN_MODEL_NAME", "qwen3.6")
_resolved_model_id: str | None = None
_resolved_model_lock = threading.Lock()

UI_HOME = Path(os.environ.get("QWEN_UI_HOME", str(Path.home() / ".qwen")))
AGENTS_DIR = UI_HOME / "agents"
SESSIONS_DIR = UI_HOME / "sessions"
UPLOADS_DIR = UI_HOME / "uploads"
GRAPH_RUNS_DIR = UI_HOME / "graph_runs"  # date-bucketed: graph_runs/YYYY-MM-DD/<run_id>.json
STATIC_DIR = Path(__file__).parent / "qwen_ui_static"

# How long to keep archived graph runs. 0 = forever. Default 30 days
# balances "see what I ran last week" with not letting the dir grow
# without bound (each run is tens of KB but a multi-graph workflow run
# 100x can put a few MB on disk).
GRAPH_RUNS_TTL_DAYS = int(os.environ.get("QWEN_UI_GRAPH_RUNS_TTL_DAYS", "30"))

# Upload constraints — stdlib http.server holds the entire body in RAM until
# we read it, so the cap protects against an accidental drag of a multi-GB
# file. 50 MB easily covers full PDFs, source trees, and most CSVs.
UPLOAD_MAX_BYTES   = int(os.environ.get("QWEN_UI_UPLOAD_MAX_BYTES", str(50 * 1024 * 1024)))
UPLOAD_MAX_TEXT    = int(os.environ.get("QWEN_UI_UPLOAD_MAX_TEXT",  "600000"))   # chars/file in chat — leaves headroom under QWEN_UI_MAX_MSG_CHARS
UPLOAD_PREVIEW     = int(os.environ.get("QWEN_UI_UPLOAD_PREVIEW",   "400"))

logger = logging.getLogger("qwen-ui")


def _ensure_dirs() -> None:
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _send_json(handler: BaseHTTPRequestHandler, status: int, body: dict | list) -> None:
    raw = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(raw)


def _send_text(handler: BaseHTTPRequestHandler, status: int, body: str,
               content_type: str = "text/plain; charset=utf-8") -> None:
    raw = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(raw)


# 25 MB ceiling on JSON bodies — a single chat message is capped at 200 KB
# (see _handle_chat_stream), but graph specs / session imports can carry
# more. Anything past 25 MB is almost certainly a misbehaving client or an
# attempted DoS, and reading it would block the worker for seconds while
# the kernel buffer drains.
_JSON_BODY_LIMIT = 25 * 1024 * 1024


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict | None:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        return None
    if length < 0 or length > _JSON_BODY_LIMIT:
        return None
    if not length:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _safe_static_path(rel: str) -> Path | None:
    """Resolve `rel` under STATIC_DIR, refusing escapes."""
    rel = rel.lstrip("/")
    target = (STATIC_DIR / rel).resolve()
    try:
        target.relative_to(STATIC_DIR.resolve())
    except ValueError:
        return None
    if not target.is_file():
        return None
    return target


def _serve_static(handler: BaseHTTPRequestHandler, rel: str) -> None:
    target = _safe_static_path(rel)
    if target is None:
        _send_text(handler, 404, "not found")
        return
    ctype, _ = mimetypes.guess_type(str(target))
    ctype = ctype or "application/octet-stream"
    data = target.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")  # local dev — never cache
    handler.end_headers()
    handler.wfile.write(data)


# /api/file — sandbox-aware file proxy used by the chat UI to inline-render
# images the model generates. Without this, browser <img src="/Users/..."
# wouldn't load at all (file:// is blocked from http:// pages). We only
# serve from a small allowlist so a hostile prompt can't ask the model
# to point at /etc/passwd.
_HOME = Path.home().resolve()
_FILE_PROXY_ROOTS: tuple[Path, ...] = tuple(
    p for p in (
        _HOME,
        Path("/tmp").resolve(),
        Path("/var/folders").resolve(),  # macOS per-user temp dirs
        Path("/private/tmp").resolve(),
        Path("/private/var/folders").resolve(),
    ) if p.exists()
)
# Image / safe-to-inline mime prefixes. Anything else returns 415 to
# discourage using this endpoint as a generic file exfil channel.
_FILE_PROXY_INLINE_PREFIXES = ("image/", "audio/", "video/", "text/")


def _is_under(root: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(root)
        return True
    except (ValueError, OSError):
        return False


# --------------------------------------------------------------------------
# file ingestion: paperclip uploads from the chat composer
# --------------------------------------------------------------------------

# Set of mime/extension prefixes treated as "text we can inline directly".
_TEXT_LIKE_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".tex",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cc",
    ".cpp", ".h", ".hpp", ".m", ".mm", ".swift", ".rb", ".php", ".pl", ".lua",
    ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".csv", ".tsv", ".xml", ".html", ".htm", ".css", ".scss", ".less",
    ".sql", ".graphql", ".proto", ".dockerfile", ".env", ".log",
}

_PDF_EXTS = {".pdf"}

# Image extensions we can OCR via tesseract.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}


def _safe_filename(name: str) -> str:
    """Strip directory components and dangerous chars. We still always
    re-prefix with a uuid before saving, but a clean filename keeps logs
    readable and the model's prose less ugly."""
    base = os.path.basename(name or "").strip()
    if not base:
        return "upload.bin"
    # Drop control chars, restrict to a sane set.
    cleaned = re.sub(r"[^A-Za-z0-9._\- ]", "_", base)[:80]
    return cleaned or "upload.bin"


def _parse_multipart(handler: BaseHTTPRequestHandler, max_bytes: int) -> tuple[list[dict], str | None]:
    """Read a multipart/form-data POST body and return [{filename, data, ctype}].
    Pure stdlib — no third-party. Returns (files, error_message)."""
    ctype = handler.headers.get("Content-Type") or ""
    if not ctype.lower().startswith("multipart/form-data"):
        return [], "expected multipart/form-data"
    # Boundary lives in the Content-Type header.
    m = re.search(r"boundary=([^\s;]+)", ctype)
    if not m:
        return [], "no boundary in Content-Type"
    boundary = m.group(1).strip().strip('"')
    if not boundary:
        return [], "empty boundary"
    try:
        clen = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        return [], "missing Content-Length"
    if clen <= 0:
        return [], "empty body"
    if clen > max_bytes:
        return [], f"body too large ({clen} > {max_bytes} bytes)"

    body = handler.rfile.read(clen)
    delim = b"--" + boundary.encode()
    parts = body.split(delim)
    files: list[dict] = []
    for part in parts:
        # Each part: \r\n<headers>\r\n\r\n<data>\r\n
        if not part or part in (b"--", b"--\r\n"):
            continue
        # Strip the leading \r\n that follows the boundary marker.
        chunk = part.lstrip(b"\r\n").rstrip(b"\r\n")
        if not chunk or chunk == b"--":
            continue
        sep = chunk.find(b"\r\n\r\n")
        if sep == -1:
            continue
        head_raw, data = chunk[:sep].decode("utf-8", "replace"), chunk[sep + 4:]
        # Drop the trailing CRLF that precedes the next boundary.
        if data.endswith(b"\r\n"):
            data = data[:-2]
        # Parse the Content-Disposition line.
        cd = ""
        ct = "application/octet-stream"
        for line in head_raw.split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                cd = line
            elif line.lower().startswith("content-type:"):
                ct = line.split(":", 1)[1].strip()
        if 'filename=' not in cd:
            continue  # not a file field
        fn_match = re.search(r'filename="([^"]*)"', cd)
        filename = fn_match.group(1) if fn_match else "upload.bin"
        if not filename:
            continue
        files.append({"filename": filename, "data": data, "ctype": ct})
    return files, None


def _extract_text_from_upload(path: Path, ext: str, ctype: str) -> tuple[str, str | None]:
    """Return (text, error). Empty text + None error means the file was
    saved but we don't know how to surface it as text (e.g. binary image).
    """
    ext = ext.lower()
    # PDF — pypdf is in the venv.
    if ext in _PDF_EXTS or "pdf" in (ctype or ""):
        try:
            import pypdf  # type: ignore
        except ImportError:
            return "", "pypdf not installed in venv"
        try:
            reader = pypdf.PdfReader(str(path))
        except Exception as e:  # noqa: BLE001
            return "", f"pdf parse failed: {type(e).__name__}: {e}"
        out: list[str] = []
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                t = ""
            t = t.strip()
            if t:
                out.append(f"--- page {i+1} ---\n{t}")
        return "\n\n".join(out).strip(), None
    # Plain text-like.
    if ext in _TEXT_LIKE_EXTS or (ctype or "").startswith("text/"):
        try:
            raw = path.read_bytes()
        except OSError as e:
            return "", f"read failed: {e}"
        # Try utf-8 first, then latin-1 as a last-resort.
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return raw.decode(enc), None
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", "replace"), None
    # Images — OCR via tesseract if available, else return a clear
    # "this is an image, the model can't see it" marker so the model
    # doesn't pretend it's a file path it can `read_file`. Image bytes
    # are stripped from the inline payload either way (qwen3.6 here is
    # text-only).
    if ext in _IMAGE_EXTS or (ctype or "").startswith("image/"):
        return _ocr_image(path)
    # Anything else — best to acknowledge it without dumping bytes.
    return "", None


def _ocr_image(path: Path) -> tuple[str, str | None]:
    """Run tesseract on `path` and return (extracted_text, error_or_None).

    Falls back to an informative "[image attachment ... model can't see
    images directly]" marker when tesseract is unavailable or yields
    nothing useful — without this, the chat prompt would carry an empty
    `<details>` block and the model would invent a `read_file` call for
    the filename, which is what the user just hit.
    """
    import shutil
    import subprocess
    tess = shutil.which("tesseract")
    if not tess:
        return (
            f"[image attachment: {path.name} — OCR unavailable on this "
            f"machine (tesseract not in PATH). The model is text-only "
            f"and can't see the image directly. Describe what you want "
            f"checked, or paste any text from the image inline.]",
            None,
        )
    try:
        # tesseract <input> stdout -- writes extracted text to stdout.
        r = subprocess.run(
            [tess, str(path), "stdout", "-l", "eng"],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return (
            f"[image attachment: {path.name} — OCR timed out after 30s. "
            f"The image was saved but its text content couldn't be "
            f"extracted. The model can't see images directly.]",
            "ocr timeout",
        )
    except Exception as e:  # noqa: BLE001
        return (
            f"[image attachment: {path.name} — OCR failed: "
            f"{type(e).__name__}: {e}. The model can't see images directly.]",
            f"ocr error: {e}",
        )
    text = (r.stdout or "").strip()
    if not text or len(text) < 8:
        # Tesseract sometimes "succeeds" with a few stray glyphs on a
        # screenshot of a UI. Below 8 chars almost always means it
        # didn't find any meaningful text — surface that clearly rather
        # than letting the model think the image is empty.
        return (
            f"[image attachment: {path.name} — saved, but OCR found "
            f"no readable text ({len(text)} chars). The model is "
            f"text-only and can't see the image content directly. "
            f"Ask the user to describe what's in it, or to paste any "
            f"text from it inline.]",
            None,
        )
    # Successful OCR — wrap with a header so the model knows this is
    # transcribed image text, not a file the user wrote.
    return (
        f"[OCR'd from image attachment: {path.name}]\n\n{text}",
        None,
    )


def _read_upload_text(upload_id: str) -> tuple[str, dict] | tuple[None, dict]:
    """Read a previously-uploaded file's extracted text from disk.
    Returns (text, meta) or (None, error_dict)."""
    if not re.fullmatch(r"[a-f0-9]{4,32}", upload_id or ""):
        return None, {"error": "invalid upload id"}
    d = UPLOADS_DIR / upload_id
    if not d.is_dir():
        return None, {"error": "no such upload"}
    files = sorted(p for p in d.iterdir() if p.is_file())
    if not files:
        return None, {"error": "upload directory empty"}
    target = files[0]
    ext = target.suffix.lower()
    text, err = _extract_text_from_upload(target, ext, "")
    return text, {"path": str(target), "filename": target.name, "ext": ext,
                  "size": target.stat().st_size, "extractor_error": err}


def _merge_attachments_into_last_user(messages: list[dict], attachments: list[dict]) -> list[dict]:
    """Find the most recent user message and prepend attachment content
    inside `<details>` blocks. Each attachment must be either a dict with
    an `id` field, OR a bare upload-id string (which we wrap into a dict
    so older/manual API callers don't silently lose their attachments).
    The model sees the full text; the UI shows a collapsed chip."""
    if not messages or not attachments:
        return messages
    # Find last user message.
    last_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_idx = i
            break
    if last_idx is None:
        return messages

    blocks: list[str] = []
    for att in attachments:
        # Accept the JS-client shape ({id, filename}) and also bare ID
        # strings, since the previous behavior silently dropped strings —
        # leaving the user's chat with no attached content and no error.
        if isinstance(att, str):
            att = {"id": att.strip()}
        if not isinstance(att, dict):
            logger.warning("ignoring non-dict attachment entry: %r", att)
            continue
        upload_id = (att.get("id") or "").strip()
        text, meta = _read_upload_text(upload_id)
        if text is None:
            blocks.append(
                f"<details><summary>📎 (failed to load attachment {upload_id!r}: "
                f"{meta.get('error')})</summary></details>"
            )
            continue
        # Re-truncate at our chat-context limit so a single huge upload doesn't
        # smother the rest of the conversation.
        if len(text) > UPLOAD_MAX_TEXT:
            text = text[:UPLOAD_MAX_TEXT] + f"\n\n…[truncated {len(text) - UPLOAD_MAX_TEXT} chars]"
        fname = meta.get("filename") or upload_id
        size = meta.get("size") or len(text)
        size_h = f"{size/1024:.1f} KB" if size >= 1024 else f"{size} B"
        chars_h = f"{len(text)} chars"
        blocks.append(
            f"<details><summary>📎 {fname} · {size_h} · {chars_h}</summary>\n\n"
            f"```\n{text}\n```\n\n</details>"
        )

    if not blocks:
        return messages

    msg = dict(messages[last_idx])
    original = msg.get("content") or ""
    msg["content"] = "\n\n".join(blocks) + ("\n\n" + original if original else "")
    messages[last_idx] = msg
    return messages


def _save_upload(file: dict) -> tuple[int, dict]:
    """Persist one file under ~/.qwen/uploads/<uuid>/<safename> and run
    the extractor. Returns (status_code, payload)."""
    raw_name = file["filename"]
    data: bytes = file["data"]
    if len(data) == 0:
        return 400, {"error": f"empty file: {raw_name}"}
    if len(data) > UPLOAD_MAX_BYTES:
        return 413, {"error": f"file too large: {raw_name} ({len(data)} bytes)"}

    safe = _safe_filename(raw_name)
    fid = uuid.uuid4().hex[:10]
    target_dir = UPLOADS_DIR / fid
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / safe
    try:
        target.write_bytes(data)
    except OSError as e:
        return 500, {"error": f"write failed: {e}"}

    ext = Path(safe).suffix.lower()
    text, err = _extract_text_from_upload(target, ext, file.get("ctype", ""))

    # Truncate the text we expose to the chat so a 1000-page PDF doesn't
    # eat the entire context window. The full extracted text remains on
    # disk in case a tool wants to read more (read_file would resolve via
    # the existing /api/file allowlist; UPLOADS_DIR lives under $HOME so
    # it's already permitted).
    full_text_chars = len(text)
    truncated = False
    if text and full_text_chars > UPLOAD_MAX_TEXT:
        text = text[:UPLOAD_MAX_TEXT] + f"\n\n…[truncated {full_text_chars - UPLOAD_MAX_TEXT} chars]"
        truncated = True

    preview = " ".join((text or "").split())[:UPLOAD_PREVIEW]

    return 200, {
        "id": fid,
        "filename": safe,
        "original": raw_name,
        "path": str(target),
        "ctype": file.get("ctype") or mimetypes.guess_type(safe)[0] or "application/octet-stream",
        "size": len(data),
        "ext": ext,
        "text": text,                 # what the model will see
        "text_chars": full_text_chars,
        "truncated": truncated,
        "preview": preview,           # short snippet for the UI chip
        "extractor_error": err,
    }


def _serve_local_file(handler: BaseHTTPRequestHandler, raw_path: str) -> None:
    """Read a local file off disk and stream it back to the browser, but
    only if it lives under one of the allowlisted roots. Used by inline
    image rendering — the LLM emits `![](/tmp/plot.png)` and the client
    rewrites the src to `/api/file?path=/tmp/plot.png`."""
    if not raw_path:
        _send_text(handler, 400, "path query param required")
        return
    try:
        p = Path(raw_path).expanduser()
    except Exception:  # noqa: BLE001
        _send_text(handler, 400, "invalid path")
        return
    if not p.exists():
        _send_text(handler, 404, "not found")
        return
    if not p.is_file():
        _send_text(handler, 400, "not a regular file")
        return
    try:
        resolved = p.resolve()
    except OSError:
        _send_text(handler, 400, "could not resolve path")
        return
    if not any(_is_under(root, resolved) for root in _FILE_PROXY_ROOTS):
        logger.warning("rejected /api/file outside allowlist: %s", resolved)
        _send_text(handler, 403, "forbidden")
        return

    ctype, _ = mimetypes.guess_type(str(resolved))
    ctype = ctype or "application/octet-stream"
    if not any(ctype.startswith(pref) for pref in _FILE_PROXY_INLINE_PREFIXES):
        # Allow only inline-friendly types. PDFs go through too (application/pdf
        # is excluded so browsers won't auto-download arbitrary binaries).
        if ctype not in ("application/pdf", "application/json"):
            _send_text(handler, 415, f"refusing to serve {ctype}")
            return
    try:
        data = resolved.read_bytes()
    except OSError as e:
        _send_text(handler, 500, f"read failed: {e}")
        return
    # Cap individual responses at 25 MB so a malicious prompt can't
    # ask the UI to slurp a giant log into the page.
    if len(data) > 25 * 1024 * 1024:
        _send_text(handler, 413, "file too large to inline (>25 MB)")
        return

    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    # Browsers love to re-fetch `<img>` on every render — let them cache.
    handler.send_header("Cache-Control", "private, max-age=300")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(data)


# --------------------------------------------------------------------------
# priority dispatcher — chat (HIGH) jumps ahead of agent cycles (NORMAL)
# --------------------------------------------------------------------------

# dflash-serve queues concurrent requests strictly FIFO at the GPU. We add
# a userland priority lock here so a chat message in the UI doesn't sit
# behind 3 agent cycles that fired moments earlier. Acquire releases as
# soon as the holder finishes its upstream call (one in-flight request
# at a time, picked by priority). Uses a Condition + heap so FIFO is
# preserved within a priority class.

PRIO_HIGH = 0
PRIO_NORMAL = 1


class PriorityDispatcher:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._heap: list[tuple[int, int, threading.Event]] = []
        self._seq = 0
        self._busy = False
        self._holder: str | None = None  # debug only
        self._depth_high = 0
        self._depth_normal = 0

    def queue_depth(self) -> dict:
        with self._cond:
            return {
                "high": self._depth_high,
                "normal": self._depth_normal,
                "busy": self._busy,
                "holder": self._holder,
            }

    def acquire(self, priority: int = PRIO_NORMAL, label: str = "") -> None:
        ev = threading.Event()
        with self._cond:
            self._seq += 1
            heapq.heappush(self._heap, (priority, self._seq, ev))
            if priority == PRIO_HIGH:
                self._depth_high += 1
            else:
                self._depth_normal += 1
            while True:
                if not self._busy and self._heap and self._heap[0][2] is ev:
                    heapq.heappop(self._heap)
                    self._busy = True
                    self._holder = label or "?"
                    if priority == PRIO_HIGH:
                        self._depth_high -= 1
                    else:
                        self._depth_normal -= 1
                    return
                self._cond.wait()

    def release(self) -> None:
        with self._cond:
            self._busy = False
            self._holder = None
            self._cond.notify_all()


DISPATCHER = PriorityDispatcher()


# --------------------------------------------------------------------------
# upstream health probe
# --------------------------------------------------------------------------

# Cache the upstream model probe — when dflash is mid-inference, even
# /v1/models can block behind a long generation (single-stream upstream),
# making /api/health hang in the browser. We refresh in the background.
_UPSTREAM_HEALTH_CACHE: dict = {"info": None, "ts": 0.0, "lock": threading.Lock()}
_UPSTREAM_HEALTH_TTL = float(os.environ.get("QWEN_UI_HEALTH_TTL", "5"))


def upstream_health() -> dict:
    """Lightweight check against /v1/models. Used by /api/health and by
    the lazy resolver below to pin the right model id for chat requests.

    Cached for a few seconds so the UI's 5-second poll loop doesn't block
    on a busy upstream — when dflash is mid-inference even /v1/models
    queues behind the active generation. The first call after a TTL miss
    is the only one that pays the latency; concurrent callers see the
    last cached value while the refresh is in flight.
    """
    cache = _UPSTREAM_HEALTH_CACHE
    now = time.monotonic()
    info = cache["info"]
    if info is not None and (now - cache["ts"]) < _UPSTREAM_HEALTH_TTL:
        return info
    url = f"{UPSTREAM_BASE}/v1/models"
    # Only one thread does the refresh; others either return cache or
    # (on cold start) wait the full timeout. Short timeout below caps
    # the worst-case latency.
    if not cache["lock"].acquire(blocking=info is None):
        # Another thread is refreshing; return cached value if any.
        return info if info is not None else {
            "upstream": "unknown", "error": "refresh in progress", "url": url,
        }
    try:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                data = json.loads(r.read())
            model_id = ""
            items = data.get("data") or []
            if items:
                model_id = items[0].get("id", "") or ""
            new_info = {"upstream": "up", "model": model_id, "url": url}
        except Exception as e:  # noqa: BLE001
            # Don't replace a known-good cache entry with a transient error;
            # mark it stale instead so the next caller will retry.
            err_info = {"upstream": "down", "error": str(e), "url": url}
            if info is not None:
                # Keep prior info if it was healthy and a recent refresh
                # might catch the upstream coming back. But still surface
                # the freshness: tag a `stale` flag for callers to inspect.
                new_info = dict(info)
                new_info["last_probe_error"] = str(e)
                new_info["stale"] = True
            else:
                new_info = err_info
        cache["info"] = new_info
        cache["ts"] = time.monotonic()
        return new_info
    finally:
        cache["lock"].release()


def _proxy_metrics() -> dict:
    url = f"{UPSTREAM_BASE}/debug/metrics"
    try:
        with urllib.request.urlopen(url, timeout=1) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "url": url}


def _optimization_metrics() -> dict:
    apc_marker = UI_HOME / ".apc_patch_installed"
    return {
        "tool_tier": {
            "rare_tools_enabled": _RARE_TOOLS_ENABLED,
            "keep_count": len(_CHAT_KEEP_TOOLS),
            "rare_count": len(_CHAT_RARE_TOOLS),
            "kept_tools": sorted(_CHAT_KEEP_TOOLS),
            "rare_tools": sorted(_CHAT_RARE_TOOLS),
        },
        "result_condense": {
            "enabled": os.environ.get("QWEN_RESULT_CONDENSE", "1")
            not in ("0", "false", "False"),
            "min_chars": int(os.environ.get("QWEN_CONDENSE_MIN_CHARS", "12000")),
            "chunk_chars": int(os.environ.get("QWEN_CONDENSE_CHUNK_CHARS", "1400")),
            "top_k": int(os.environ.get("QWEN_CONDENSE_TOP_K", "5")),
            "lead_chars": int(os.environ.get("QWEN_CONDENSE_LEAD_CHARS", "1200")),
        },
        "runtime": {
            "repetition_penalty": float(os.environ.get("DFLASH_REP_PENALTY", "0.0") or "0.0"),
            "repetition_history": int(os.environ.get("DFLASH_REP_HISTORY", "64") or "64"),
            "lazy_draft_eval": os.environ.get("DFLASH_LAZY_DRAFT_EVAL", "1") == "1",
            "prompt_cache": os.environ.get("DFLASH_PROMPT_CACHE", "on"),
            "prompt_cache_min": int(os.environ.get("DFLASH_PROMPT_CACHE_MIN", "256") or "256"),
            "prompt_cache_slots": int(os.environ.get("DFLASH_PROMPT_CACHE_SLOTS", "4") or "4"),
            "prompt_cache_gb": float(os.environ.get("DFLASH_PROMPT_CACHE_GB", "0") or "0"),
            "prefill_chunk": int(os.environ.get("DFLASH_PREFILL_CHUNK", "2048") or "2048"),
            "verify_len": int(os.environ.get("DFLASH_VERIFY_LEN", "0") or "0"),
            "quantize_kv": os.environ.get("DFLASH_QUANTIZE_KV", "0"),
            "apc_patch_marker": str(apc_marker),
            "apc_patch_installed": apc_marker.exists(),
        },
        "proxy": _proxy_metrics(),
    }


def resolve_model_id() -> str:
    """Return the actual id loaded by upstream. Cached after first lookup;
    fall back to the env alias if upstream is unreachable so callers still
    get a sane request body (the upstream will then return its own error)."""
    global _resolved_model_id
    if _resolved_model_id:
        return _resolved_model_id
    with _resolved_model_lock:
        if _resolved_model_id:  # double-check under lock
            return _resolved_model_id
        info = upstream_health()
        if info.get("upstream") == "up" and info.get("model"):
            _resolved_model_id = info["model"]
            logger.info("resolved model id: %s (alias: %s)", _resolved_model_id, MODEL_ALIAS)
            # Propagate to agent_tools so its internal upstream calls
            # (compaction, explore subagent, graph nodes) use the same
            # exact id — the proxy enforces an exact-match check and
            # would 400 otherwise. agent_tools reads this lazily via
            # os.environ.get("QWEN_MODEL_NAME"), so setting it here
            # works even before agent_tools is imported.
            os.environ["QWEN_MODEL_NAME"] = _resolved_model_id
        else:
            _resolved_model_id = MODEL_ALIAS
            logger.warning("could not resolve model id (%s); using alias '%s'",
                           info.get("error"), MODEL_ALIAS)
        return _resolved_model_id


# --------------------------------------------------------------------------
# chat: streaming with backend tool-call loop
# --------------------------------------------------------------------------

# The "context window" surfaced to the chat UI is the threshold for auto-
# compaction. The model's full 256k context is real, but on this hardware
# decode TPS drops sharply past ~30k tokens (340 tok/s @14k → 80 tok/s
# @43k), so we trigger compaction at 40k by default — well before the
# slowdown becomes painful. Tunable via QWEN_UI_CONTEXT_TOKENS for users
# who want longer context retention at the cost of slower turns.
CONTEXT_WINDOW_TOKENS = int(
    os.environ.get("QWEN_UI_CONTEXT_TOKENS")
    or os.environ.get("QWEN_AGENT_COMPACT_AT")
    or "40000"
)

CHAT_MAX_TURNS = int(os.environ.get("QWEN_UI_MAX_TURNS", "100"))
# Hard cap on total tool calls within a single chat turn. The cache-hit loop
# detector (CHAT_LOOP_BREAK_THRESHOLD) fires only after the model starts
# RE-asking, which can be 50+ tool calls into a runaway. This second cap
# stops the model dead at N tool calls regardless of cache state — raised
# 30 → 100 to support long-horizon code-reading tasks (e.g. SWE-bench Pro
# repo navigation) where 30 tool calls is too tight.
CHAT_MAX_TOOL_CALLS = int(os.environ.get("QWEN_UI_MAX_TOOL_CALLS", "100"))
# After N consecutive turns where EVERY tool call was a cache hit, treat the
# model as genuinely stuck and inject a hard "synthesize and stop" nudge.
# Mirrors agent.py CLI behavior. Tunable via env so the threshold can be
# tightened without a redeploy.
CHAT_LOOP_BREAK_THRESHOLD = int(os.environ.get("QWEN_UI_LOOP_BREAK", "3"))
# Bumped 16384 (was 4096) — Qwen3.6's thinking mode routinely emits 5-10k
# tokens of <think> content alone before the actual response, so 4096 was
# truncating thinking-heavy answers mid-LaTeX / mid-code. Abandonment is
# already handled by the `is_alive()` check in the stream loop (client
# disconnect aborts decoding), so a higher cap doesn't waste GPU on closed
# tabs. Tunable via QWEN_UI_MAX_TOKENS — drop to 4096 if your workload is
# always non-thinking conversational replies.
CHAT_MAX_TOKENS = int(os.environ.get("QWEN_UI_MAX_TOKENS", "16384"))
# Outer safety net for tool output. Tools have their own per-result caps in
# agent_tools.py (web_fetch, bash, etc), so this is the upper limit AFTER
# those run. Bumped from 8 KB → 250 KB so a fetched arxiv page doesn't get
# clipped to ~2 K tokens by the UI when the model has 256 K context to work
# with.
TOOL_RESULT_TRUNC = int(os.environ.get("QWEN_UI_TOOL_TRUNC", "250000"))


def _today_str() -> str:
    return _dt.date.today().strftime("%A, %B %-d, %Y")


# Hand-curated terse tool blurbs for the chat tab. Each replaces a 100-150
# token verbose description with a 5-15 token one — saving ~5k tokens of
# baseline context. The model gets the FULL parameter schema from the
# `parameters` field; only the prose description is trimmed. We test that
# tool selection still works after each change.
_CHAT_TOOL_BLURBS: dict[str, str] = {
    # filesystem
    "list_files":      "list directory contents (names + sizes).",
    "read_file":       "read file contents (use offset/limit to slice large files).",
    "grep":            "regex search across files; returns matching lines + paths.",
    "write_file":      "write a whole file (use apply_patch for edits to existing files).",
    "write_file_verified": "write file then run a Python verifier; revert on failure.",
    "edit_file":       "search-and-replace within a file (one-shot string swap).",
    "apply_patch":     "apply a unified diff (preferred for edits to existing files).",
    "append_finding":  "append a `## heading` section to a markdown artifact.",
    # exec
    "bash":            "run shell/build commands; avoid for Python snippets or pytest.",
    "python_run":      "run Python snippets/data work in a persistent kernel; not shell.",
    "python_reset":    "reset the persistent Python kernel (clears all variables).",
    "test_run":        "run pytest/unittest-style checks; prefer over bash for tests.",
    "notebook_edit":   "edit a Jupyter notebook cell.",
    "notebook_run":    "run a Jupyter notebook end-to-end.",
    # web
    "web_search":      "DuckDuckGo search; returns title/url/snippet for top results.",
    "web_fetch":       "fetch a URL and return readable text/markdown (handles JS).",
    "web_outline":     "fetch a URL and return only the heading hierarchy (h1-h4).",
    # explore / subagent
    "explore":         "spawn a read-only subagent for open-ended questions.",
    "subagent_implement": "spawn a read+write subagent for a self-contained task.",
    # memory
    "memory_save":     "save a long-term memory (persists across chats); key + content + tags.",
    "memory_get":      "fetch one memory by exact key.",
    "memory_search":   "search saved memories by semantic similarity + keyword.",
    "memory_list":     "list recent memory keys.",
    "memory_delete":   "delete a memory entry by key.",
    # task tracking
    "todo_write":      "maintain a structured todo list for this session.",
    # worktree
    "enter_worktree":  "create an isolated git worktree and switch into it.",
    "exit_worktree":   "leave the current worktree and clean up.",
    # time / misc
    "now":             "current time (optionally in a named timezone like 'America/New_York').",
    "csv_summary":     "describe a CSV file: shape, column dtypes, head rows, numeric stats.",
    "inspect_data":    "auto-summarize a data file (CSV/TSV/JSON/JSONL); shape + sample.",
    "github_repo":     "GitHub public API; action=info|list|read|readme for repo metadata, dir listings, file contents.",
    "arxiv_search":    "search arXiv; returns each paper's title + authors + arXiv id + abstract excerpt.",
    "arxiv_fetch":     "fetch one arXiv paper by id or URL; what=abstract|html|pdf.",
    "arxiv_get":       "fetch arXiv paper metadata + abstract.",  # legacy alias
    "pdf_extract":     "extract text from a PDF (local file path OR https URL).",
    "doi_resolve":     "resolve a DOI to formatted citation metadata (year, authors, journal, title).",
    "sec_filings":     "list recent SEC EDGAR filings for a US-listed company; returns direct URLs to 10-K/10-Q/8-K/etc. Use BEFORE web_search whenever you need an authoritative SEC document.",
    "make_table":      "render a deterministic Markdown table from headers + rows.",
    "mcp_list":        "list custom MCP servers + their tools.",
    "agent_graph_list": "list available multi-agent graphs.",
    "agent_graph_run": "run a multi-agent graph; takes graph name + initial input.",
    "done":            "signal task complete (FINAL tool call); takes a 1-2 sentence summary.",
}


_ULTRA_TERSE = os.environ.get("QWEN_UI_ULTRA_TERSE", "1") in ("1", "true", "True")

# Chat-only tool tier. The current target is not the smallest possible list;
# it is the least-confusing startup list. Keep specialized tools when they
# route the model away from wasteful generic calls (e.g. test_run instead of
# bash pytest, github_repo instead of web_fetch GitHub HTML, web_outline
# instead of fetching an entire page just to inspect structure).
_CHAT_KEEP_TOOLS = frozenset({
    "bash", "read_file", "list_files", "grep",
    "edit_file", "write_file", "apply_patch",
    "test_run",
    "web_search", "web_fetch", "web_outline",
    "github_repo", "arxiv_search", "arxiv_fetch", "doi_resolve", "sec_filings",
    "now",
    "mcp_list",
    # memory: save (write) + search (read) + delete (hygiene). get/list are
    # redundant with search since search returns content. Without delete,
    # stale memories accumulate and search returns more bad hits over time.
    "memory_save", "memory_search", "memory_delete",
    # scratchpad covers transient in-task notes (no memory pollution);
    # ask_user gives the model a clean exit from ambiguous prompts so it
    # doesn't loop on guessing; graph_compose lets a chat session escalate
    # into a graph mid-conversation without context-switching to the
    # graph-designer panel.
    "scratchpad", "ask_user", "graph_compose",
    "pdf_extract", "make_table", "python_run", "inspect_data",
    "agent_graph_list", "agent_graph_run",
})

_CHAT_RARE_TOOLS = frozenset({
    # csv_summary overlaps inspect_data; keep one data inspector in chat.
    "csv_summary",
    # subagent dispatch — still easy to misuse from interactive chat:
    "explore", "subagent_implement",
    # memory get/list — search covers both, see comment above:
    "memory_get", "memory_list",
    # autonomous-loop affordances — chat ends naturally:
    "done", "todo_write",
    "append_finding", "write_file_verified",
    # heavyweight or duplicative of bash:
    "enter_worktree", "exit_worktree",
    "notebook_edit", "notebook_run",
    "python_reset",
    # MCP register/unregister — UI panel owns this lifecycle; mcp_list stays:
    "mcp_register", "mcp_unregister",
})
_RARE_TOOLS_ENABLED = os.environ.get("QWEN_UI_RARE_TOOLS", "0") in ("1", "true", "True")


def _chat_tool_tier(tools: list[dict]) -> list[dict]:
    """Filter the tool list down to the chat-tier subset unless the user
    opts in to rare tools. Saves ~1.2 KB / ~300 tokens per chat turn.

    Uses the keep-list rather than just the rare-list so newly-added tools
    don't silently leak into chat — they have to be opted in by name.
    Registered MCP tools (named `mcp_<server>__<tool>`, double-underscore)
    are always included since the user opted into them by registering."""
    if _RARE_TOOLS_ENABLED:
        return tools
    keep = []
    for t in tools:
        name = (t.get("function") or {}).get("name", "")
        if name in _CHAT_KEEP_TOOLS:
            keep.append(t)
        elif name.startswith("mcp_") and "__" in name:
            keep.append(t)  # user-registered MCP tool
    return keep


def _terse_tools(tools: list[dict]) -> list[dict]:
    """Return a copy of `tools` with descriptions AND per-parameter prose
    trimmed for the chat tab. Two tiers:

      - default ("ultra terse", `QWEN_UI_ULTRA_TERSE=1`): drop ALL parameter
        `description` fields. The model figures out parameters from their
        names plus the one-line tool blurb. Saves ~5 KB / ~1300 tokens vs
        the regular terse mode on the current 42-tool schema. Set
        `QWEN_UI_ULTRA_TERSE=0` to revert if a tool turns out to need its
        per-param prose to be invoked correctly.
      - regular: shrink each param `description` to its first clause (≤80
        chars). The mode used historically.

    The tool-level blurb (from `_CHAT_TOOL_BLURBS` or first sentence of the
    full description) is preserved either way — that's the load-bearing
    signal for the model to pick the right tool.
    """
    out: list[dict] = []
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            out.append(t); continue
        fn = (t.get("function") or {}).copy()
        name = fn.get("name") or ""
        blurb = _CHAT_TOOL_BLURBS.get(name)
        if blurb is None:
            desc = (fn.get("description") or "").strip()
            head = desc.split(". ", 1)[0]
            if len(head) > 120:
                head = head[:120].rstrip() + "…"
            blurb = (head + ".") if head and not head.endswith(".") else head
        fn["description"] = blurb
        params = fn.get("parameters")
        if isinstance(params, dict):
            props = params.get("properties")
            if isinstance(props, dict):
                new_props = {}
                for pname, pspec in props.items():
                    if isinstance(pspec, dict):
                        ps = dict(pspec)
                        if _ULTRA_TERSE:
                            ps.pop("description", None)
                            ps.pop("default", None)  # runtime applies them
                            ps.pop("examples", None)
                        else:
                            d = ps.get("description")
                            if isinstance(d, str) and len(d) > 80:
                                cut = re.split(r"\.\s|—|;", d, 1)[0].strip()
                                if len(cut) > 80:
                                    cut = cut[:80].rstrip() + "…"
                                ps["description"] = cut + ("." if cut and not cut.endswith(".") else "")
                        new_props[pname] = ps
                    else:
                        new_props[pname] = pspec
                params2 = dict(params); params2["properties"] = new_props
                fn["parameters"] = params2
        new_t = dict(t); new_t["function"] = fn
        out.append(new_t)
    return out


# Static prefix is byte-identical across calls (cacheable at every level:
# dflash prompt-cache, OS page-cache, KV-prefix). Dynamic bits (date, cwd)
# live in the SUFFIX so the prefix stays hot across midnight rollovers and
# cwd switches.
#
# Compact chat prompt preserving the headless agent's core guardrails:
# tool-use guidance, anti-loop discipline, quantitative metadata checks,
# no-preamble behavior, fenced code blocks, and ASCII-LaTeX for math.
_CHAT_SYSTEM_STATIC = (
    "Concise. Call tools when needed, else answer directly. No preambles, no "
    "'Sure'/'Certainly' openers. Code in fenced blocks.\n"
    "Before your first tool call, state in one sentence what you're about to "
    "do — the user can't see tool calls. Don't narrate deliberation. "
    "End-of-turn summary: 1-2 sentences, what changed and what's next.\n"
    "Match the user's requested output format exactly — no extra columns, "
    "fields, debug output, or commentary they didn't ask for.\n"
    "Use one short planning pass, then act. If uncertain, convert the "
    "uncertainty into one concrete check; after that check, proceed, answer "
    "with caveats, or stop. Repeated self-questioning is a stop signal. "
    "If the request is genuinely ambiguous in a way that changes the "
    "deliverable, call `ask_user(question, options)` rather than guessing.\n"
    "Graphs and subagents: when the task decomposes into research → analyze "
    "→ produce, call `agent_graph_list` to see existing graphs and "
    "`agent_graph_run` to launch one — each node has its own context, so "
    "this is the right move for multi-step pipelines that would otherwise "
    "blow the window. `graph_compose(description)` builds a fresh graph if "
    "none fit. Trust but verify: read the artifact a subagent claims to "
    "have written before forwarding the claim.\n"
    "`[cached…]` means the same evidence is already available; use it or "
    "change a real dimension, not wording. `[REFUSED…]` is final for that "
    "tool path — synthesize from above; if the answer isn't there, say so "
    "and stop. Don't loop. "
    "An `[empty: …]` web_fetch is a dead host (paywall/JS wall); stop "
    "retrying that domain and try a different source.\n"
    "Web search discipline: before each `web_search`, identify the new "
    "dimension (entity, period, metric, source type, site/domain, filetype, "
    "geography, exact phrase). Synonyms, word order, filler words, or "
    "restating the question are not new dimensions. Do not parallel-search "
    "the same data point. After near-duplicate/cached/empty/refused results, "
    "fetch a promising result, use a specialized tool or `find_in_url`, or "
    "synthesize.\n"
    "Comments: default to writing none. Only when the WHY is non-obvious "
    "(hidden constraint, subtle invariant, surprising behavior). Never "
    "explain WHAT well-named code does, and never reference the current "
    "task/fix/callers — those belong in the PR description.\n"
    "Reversibility: local edits/reads — proceed freely. Hard-to-reverse or "
    "shared-state actions (`git push`, force-push, dropping tables, "
    "modifying CI, `--no-verify`) — confirm with the user before each one, "
    "even if a similar action was approved earlier.\n"
    # Synced from scripts/agent.py SYSTEM_PROMPT_STATIC. Keep these two
    # blocks (Quantitative answers + Sibling-metric ambiguity) in sync
    # across the two surfaces so the chat tab and headless agent give
    # the same single-shot reliability. Generalized iter 30: rules apply
    # across filings, market data, returns, macro — not just SEC.
    "Quantitative answers: parse (entity, period, metric, units, scope) "
    "BEFORE retrieving and match each exactly — applies to filings, "
    "prices, returns, and macro alike. Quarter → quarterly source; "
    "intraday → intraday; calendar year ≠ fiscal year; YTD ≠ TTM. "
    "Default period when none given = latest completed: most recent "
    "annual report for fundamentals, prior trading session's close for "
    "prices, latest released print for macro. Closely-related metrics "
    "are not interchangeable: 'long-term' ≠ 'all'; 'GAAP' ≠ 'non-GAAP'; "
    "'diluted' ≠ 'basic'; 'last trade' ≠ 'mid' ≠ 'official close'; "
    "'total return' ≠ 'price return'. Re-read the row/column label "
    "before quoting. After producing a numeric answer, run a 1-line "
    "self-audit: what entity/period/units/scope/source does each "
    "figure correspond to? Fix or confirm before finalizing.\n"
    # Synced from agent.py sibling-metric rule. Generalized: convention-
    # vs-literal ambiguity isn't unique to SEC line items.
    "Sibling-metric ambiguity: when a term maps to ≥2 plausible "
    "candidates and convention disagrees with the literal reading, "
    "name the one you used and give the alternative. Examples: 'all "
    "debt' (literal: total) vs long-term-only (convention for "
    "refinancing); 'price' (last/mid/close/VWAP); 'return' (total/"
    "price/log/simple); 'volatility' (realized/implied). Format: "
    "'<label used> = <value>; (alt: <other label> = <other value>)'.\n"
    "Math: LaTeX only (`$..$` inline, `$$..$$` display) with ASCII commands "
    "(`\\alpha`, `\\geq`, `\\sum`, `\\frac{a}{b}`). Never raw Unicode glyphs.\n"
)


def _chat_system_prompt(cwd: str) -> str:
    """Static prefix + short dynamic tail. The prefix is byte-identical
    every request so dflash's prompt cache hits; only the tail (date + cwd,
    a couple hundred chars) misses on date rollover or cwd change."""
    return (
        _CHAT_SYSTEM_STATIC
        + "\n# Session\n"
        + f"Today: {_today_str()}. "
        + "Today is only temporal context. For real-world latest/current/today "
        + "facts, verify with a current source unless the answer is purely "
        + "local/session state. For web searches include the year in the query "
        + "(`AI news 2026`, not `AI news`). Use `now` only for timezone math.\n"
        + f"Cwd: {cwd}.\n"
    )


# Slash-command help rendered on `/help`. Module-level so the test harness
# (which exercises the routing on a duck-typed handler) can reach it.
_SLASH_HELP = (
    "**Chat slash commands** — these run server-side without invoking the model:\n\n"
    "- `/help` — show this help\n"
    "- `/graphs` — list available agent graphs (compact table)\n"
    "- `/graph <name> [json-inputs]` — run a graph; events stream inline\n"
    "  e.g. `/graph market_research {\"topic\": \"US equities today\"}`\n"
    "- `/graph_compose <description>` — design + save a fresh graph from a natural-language paragraph\n"
    "- `/scratchpad [read|clear] [key]` — show/clear in-task notes (default: read default pad)\n"
    "- `/tools` — show the chat-tier tool list (one line each)\n\n"
    "Anything not starting with `/` is sent to the model normally. "
    "Slash commands are also still available **inside** a model response: "
    "the model itself can call `agent_graph_run`, `graph_compose`, `scratchpad`, etc."
)


def _sse_frame(event: str, data: dict) -> bytes:
    """Build a single SSE frame. Events use \\n\\n separator per spec."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


# Pattern that detects an in-progress <tool_call> opening in streamed content.
# We use this to decide whether to emit content deltas to the browser or
# buffer them until the matching closer arrives.
_TC_OPEN = "<tool_call>"
_TC_CLOSE = "</tool_call>"


def _strip_tool_calls_for_history(content: str) -> str:
    """Remove `<tool_call>...</tool_call>` blocks from text we want to keep
    in the assistant `content` field — those go in `tool_calls` instead."""
    return re.sub(r"<tool_call>.*?</tool_call>\s*", "",
                  content, flags=re.DOTALL).strip()


# --------------------------------------------------------------------------
# real token counting (HF tokenizer)
# --------------------------------------------------------------------------
# dflash-mlx doesn't return `usage` chunks even when you pass
# `stream_options.include_usage`, so the client's context bar would either
# stay drifty (char-based estimate) or require an extra non-streaming
# completion just to learn the prompt size. Instead we load the model's
# real tokenizer here and count locally — same algorithm dflash would use,
# free, ~10ms per call.

_tokenizer_state: dict = {"obj": None, "loaded": False, "name": None}
_tokenizer_lock = threading.Lock()


def _get_tokenizer():
    """Lazy-load the chat model's tokenizer. Returns None if unavailable
    (no transformers, model dir missing, etc) — caller falls back to the
    char-based estimate."""
    if _tokenizer_state["loaded"]:
        return _tokenizer_state["obj"]
    with _tokenizer_lock:
        if _tokenizer_state["loaded"]:
            return _tokenizer_state["obj"]
        _tokenizer_state["loaded"] = True
        # Heuristic order: env override → resolved model id (a path) →
        # MODEL_ALIAS → typical default. resolve_model_id() returns the
        # exact id dflash advertises, which for OptiQ is a relative path
        # like ./models/Qwen3.6-35B-A3B-OptiQ-4bit — that's also where
        # tokenizer.json lives, so AutoTokenizer can find it.
        candidates: list[str] = []
        env_dir = os.environ.get("QWEN_MODEL_DIR") or os.environ.get("QWEN_TOKENIZER_DIR")
        if env_dir:
            candidates.append(env_dir)
        candidates.append(resolve_model_id())
        candidates.append(MODEL_ALIAS)
        seen: set[str] = set()
        for name in candidates:
            if not name or name in seen:
                continue
            seen.add(name)
            # Make relative paths resolvable from the project root.
            if not os.path.isabs(name) and not name.startswith("./") and not "/" in name:
                # bare HF id — try as-is
                pass
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                if tok is not None:
                    _tokenizer_state["obj"] = tok
                    _tokenizer_state["name"] = name
                    logger.info("tokenizer loaded: %s", name)
                    break
            except Exception as e:  # noqa: BLE001
                logger.debug("tokenizer load %r failed: %s", name, e)
                continue
        if _tokenizer_state["obj"] is None:
            logger.warning(
                "could not load any tokenizer — context bar will fall back "
                "to a char-based estimate"
            )
        return _tokenizer_state["obj"]


def _normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    """Qwen3.6's chat template is strict about message shape:
      - tool_calls[i].function.arguments must be a dict (NOT a JSON string).
      - assistant `content` must be a non-None string ("" is fine).
    The OpenAI wire format uses a JSON-string for arguments, which breaks
    Jinja's `arguments | items` lookup with `Can only get item pairs from
    a mapping`. Normalize once before counting / persisting."""
    out: list[dict] = []
    for m in messages:
        m2 = dict(m)
        # Ensure assistant content is a string, never None.
        if m2.get("role") == "assistant" and m2.get("content") is None:
            m2["content"] = ""
        # Convert tool_calls[].function.arguments from str → dict.
        tcs = m2.get("tool_calls")
        if isinstance(tcs, list):
            new_tcs = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                fn = (tc.get("function") or {}).copy()
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        # Leave it as a string-wrapped dict so the template
                        # at least doesn't crash on `items`.
                        fn["arguments"] = {"_raw": args}
                tc2 = dict(tc)
                tc2["function"] = fn
                new_tcs.append(tc2)
            m2["tool_calls"] = new_tcs
        out.append(m2)
    return out


def _count_chat_tokens(messages: list[dict], tools: list[dict] | None = None) -> int | None:
    """Render `messages` through the model's chat template and return the
    exact prompt-token count. None on failure.

    Delegates to `agent_tools.real_tokens` so the agent CLI, graph nodes,
    chat compaction, and this status-bar count all share ONE tokenizer
    instance and ONE counting algorithm — they're guaranteed to agree."""
    try:
        from agent_tools import real_tokens, _load_tokenizer
    except Exception:  # noqa: BLE001
        return None
    if _load_tokenizer() is None:
        return None
    try:
        return real_tokens(messages, tools)
    except Exception as e:  # noqa: BLE001
        logger.debug("token count failed: %s", e)
        return None


def _quick_upstream_alive(timeout: float = 2.5) -> tuple[bool, str]:
    """Fast probe: is the upstream actually answering, or is the worker hung?
    Returns (alive, error_msg). The HTTP-level health check distinguishes
    'process exists but doesn't respond' (the failure mode that produces
    [Errno 60] timeouts after 60s) from 'normal upstream temporarily busy'.
    """
    try:
        req = urllib.request.Request(
            f"{UPSTREAM_BASE}/v1/models",
            headers={"User-Agent": "qwen-ui/preflight"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read(64)  # don't care about contents, just confirm a response
        return True, ""
    except urllib.error.HTTPError as e:
        # An HTTP error means the worker IS responding (just sad), so the
        # subsequent /chat/completions will fail with a useful message.
        return True, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"unreachable: {e.reason}"
    except (TimeoutError, OSError) as e:
        return False, f"timed out: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _post_upstream_stream(messages: list[dict], tools: list[dict],
                          enable_thinking: bool):
    """Open an SSE connection to upstream's /v1/chat/completions. Returns
    the urllib response object — caller iterates lines and parses. Caller
    is responsible for closing.

    Does a fast pre-flight check first so a hung worker fails in 2.5s with
    a clear error instead of letting the user wait for a 60-second TCP
    timeout. The pre-flight is *cheap* (`/v1/models` is essentially a
    static handler), and it answers the actual question we care about:
    'is dflash-serve responding to anything?'.
    """
    alive, err = _quick_upstream_alive()
    if not alive:
        # Build a urllib-style error so the caller's existing handling
        # path (which catches URLError) takes over with a useful message.
        raise urllib.error.URLError(
            f"inference worker not responding ({err}). "
            f"Run `qwen restart` in a terminal to bring it back."
        )

    body: dict = {
        "model": resolve_model_id(),
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "stream": True,
        "stream_options": {"include_usage": True},   # final chunk carries
                                                     # prompt_tokens etc.
        "max_tokens": CHAT_MAX_TOKENS,
    }
    if not enable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    # Optional: dump the upstream body to /tmp for diagnostics. Off by
    # default; set QWEN_UI_DEBUG_UPSTREAM_BODY=1 to enable. Helped diagnose
    # an orphan-process state bug where two qwen_ui.py PIDs were both
    # bound to port 8001 and probes were hitting the stale one.
    if os.environ.get("QWEN_UI_DEBUG_UPSTREAM_BODY"):
        try:
            import hashlib
            digest = hashlib.md5(json.dumps(body, sort_keys=True,
                                             ensure_ascii=False).encode()).hexdigest()[:8]
            dump_path = f"/tmp/qwen_ui_upstream_{digest}.json"
            with open(dump_path, "w") as fh:
                json.dump(body, fh, indent=2, default=str, ensure_ascii=False)
            logger.info("upstream body dumped to %s", dump_path)
        except Exception:  # noqa: BLE001
            pass
    req = urllib.request.Request(
        f"{UPSTREAM_BASE}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # 600s is the per-read timeout — enough for the model to take its time
    # generating tokens, but if the connection completely silences for that
    # long the user has long since given up. Lowered slightly so an aborted
    # client doesn't have us waiting 10 minutes on a dead socket.
    return urllib.request.urlopen(req, timeout=300)


_THINK_CLOSE = "</think>"
_IM_END = "<|im_end|>"


# Lowercased phrases that almost never benefit from a thinking block. The
# model already responds well to these instantly, and skipping `<think>`
# saves 200-500 tokens per turn (~1-2s wall on Qwen3.6).
_TRIVIAL_REPLIES = {
    "ok", "okay", "k", "yes", "y", "no", "n", "yep", "nope",
    "thanks", "thank you", "ty", "tysm", "great", "cool", "nice",
    "go", "go ahead", "do it", "sure", "please do", "continue",
    "perfect", "got it", "makes sense", "understood",
}
# Compact regex for "imperative single tool call" — these are usually
# crystal-clear about what to do; thinking adds nothing.
_TOOL_IMPERATIVE = re.compile(
    r"^(call|run|use|fetch|get|search|open|read|list|check|show)\s+"
    r"[\w./@:-]+\s*(\(|$)", re.IGNORECASE)


def _is_trivial_user_msg(text: str) -> bool:
    """Heuristic: should this user message skip the thinking block?

    Matches:
      - Tiny acknowledgements / yes-no replies (≤ 30 chars after lower).
      - Short imperatives that name a tool/url ("fetch https://...").
      - Bare URLs ("https://arxiv.org/abs/2401.12345" — model fetches it).
    Misses (correctly):
      - Genuine questions, multi-clause prompts, anything > 80 chars.
    """
    if not text:
        return False
    t = text.strip()
    if len(t) > 80:
        return False
    low = t.lower().rstrip(".!?")
    if low in _TRIVIAL_REPLIES:
        return True
    if _TOOL_IMPERATIVE.match(t):
        return True
    # Bare URL — most likely "fetch this".
    if re.fullmatch(r"https?://\S+", t):
        return True
    return False


def _emit_visible(buf: str, in_tool_call: bool, emit) -> tuple[str, bool]:
    """Walk `buf` searching for <tool_call>…</tool_call> boundaries; emit
    anything outside as `message` events; suppress anything inside (those
    get re-emitted as structured tool_call events after parsing). Holds
    back the last few chars when a partial open-tag is possible. Returns
    (remaining_buffer, new_in_tool_call_state)."""
    while buf:
        if in_tool_call:
            idx = buf.find(_TC_CLOSE)
            if idx == -1:
                return "", True
            buf = buf[idx + len(_TC_CLOSE):]
            in_tool_call = False
        else:
            idx = buf.find(_TC_OPEN)
            if idx == -1:
                # hold back enough chars to detect a partial open tag
                keep = max(0, len(buf) - (len(_TC_OPEN) - 1))
                if keep > 0:
                    chunk = buf[:keep]
                    buf = buf[keep:]
                    chunk = chunk.replace(_IM_END, "")
                    if chunk:
                        emit("message", {"delta": chunk})
                return buf, False
            else:
                pre = buf[:idx].replace(_IM_END, "")
                if pre:
                    emit("message", {"delta": pre})
                buf = buf[idx + len(_TC_OPEN):]
                in_tool_call = True
    return "", in_tool_call


def _stream_one_completion(messages, tools, enable_thinking, emit, is_alive=lambda: True):
    """Run one upstream completion. Streams `thinking`/`message` events to
    the browser. Returns (assistant_msg, tool_calls) where assistant_msg is
    OpenAI-shape and tool_calls is the parsed list (possibly empty).

    Qwen3.6's chat template prepends `<think>` when thinking is enabled,
    so the upstream emits reasoning text and then `</think>` to mark the
    boundary. We treat content before that closer as `thinking` events
    and content after as `message` events. When thinking is disabled,
    we go straight to message mode.
    """
    content_parts: list[str] = []
    structured_tool_calls: list[dict] = []
    in_tool_call = False
    in_thinking = bool(enable_thinking)
    pending_msg_buf = ""    # buffer for the message/tool-call boundary search
    pending_think_buf = ""  # buffer while still inside reasoning, scanning for </think>
    # Track upstream's finish_reason so we can tell the user when their
    # response was truncated by max_tokens vs. ended naturally.
    final_finish_reason: str | None = None

    DISPATCHER.acquire(PRIO_HIGH, label="chat")
    try:
        resp = _post_upstream_stream(messages, tools, enable_thinking)
    except Exception:
        DISPATCHER.release()
        raise

    try:
        for raw in resp:
            # Bail out the moment the client is gone — otherwise dflash
            # keeps decoding to max_tokens and the upstream socket stays
            # held even though nobody's reading.
            if not is_alive():
                logger.info("client gone — aborting upstream stream")
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            # Forward upstream's usage stats — the final non-delta chunk
            # carries `prompt_tokens`/`completion_tokens` when stream_options.
            # include_usage=True. The client snaps its estimator to this
            # exact count, so the context-window bar stops drifting.
            if obj.get("usage"):
                emit("usage", obj["usage"])
            choices = obj.get("choices") or []
            if not choices:
                continue
            # Capture finish_reason on the final chunk — "length" means the
            # model hit max_tokens and the response is truncated.
            fr = choices[0].get("finish_reason")
            if fr:
                final_finish_reason = fr
            delta = choices[0].get("delta") or {}
            streamed_tcs = delta.get("tool_calls")
            if isinstance(streamed_tcs, list):
                for tc in streamed_tcs:
                    if isinstance(tc, dict) and (tc.get("function") or {}).get("name"):
                        structured_tool_calls.append(tc)

            # Some servers (post-proxy) emit reasoning_content as a
            # separate field. Treat that as thinking unconditionally.
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                emit("thinking", {"delta": r})

            c = delta.get("content")
            if not c:
                continue
            content_parts.append(c)

            if in_thinking:
                pending_think_buf += c
                # Look for the closer; emit prefix as thinking, switch to message.
                idx = pending_think_buf.find(_THINK_CLOSE)
                if idx == -1:
                    # No closer yet — emit as thinking but hold back enough
                    # chars to detect a split closer (e.g. "</thi" + "nk>").
                    keep = max(0, len(pending_think_buf) - (len(_THINK_CLOSE) - 1))
                    if keep > 0:
                        emit("thinking", {"delta": pending_think_buf[:keep]})
                        pending_think_buf = pending_think_buf[keep:]
                    continue
                # Closer found.
                pre_close = pending_think_buf[:idx]
                if pre_close:
                    emit("thinking", {"delta": pre_close})
                # The text after </think> is the start of the visible answer.
                tail = pending_think_buf[idx + len(_THINK_CLOSE):]
                pending_think_buf = ""
                in_thinking = False
                pending_msg_buf += tail.lstrip("\n")
            else:
                pending_msg_buf += c

            if not in_thinking and pending_msg_buf:
                pending_msg_buf, in_tool_call = _emit_visible(
                    pending_msg_buf, in_tool_call, emit
                )
    finally:
        try:
            resp.close()
        finally:
            DISPATCHER.release()

    # Flush any held-back tails.
    if in_thinking and pending_think_buf.strip():
        emit("thinking", {"delta": pending_think_buf})
    if pending_msg_buf and not in_tool_call:
        tail = pending_msg_buf.replace(_IM_END, "")
        if tail.strip():
            emit("message", {"delta": tail})

    full_content = "".join(content_parts)
    visible_text, _think = split_reasoning(full_content)
    visible_text = _strip_tool_calls_for_history(visible_text).replace(_IM_END, "").strip()
    tool_calls = structured_tool_calls or parse_tool_calls(full_content)

    # Tell the user when output was capped by max_tokens — otherwise it
    # looks like the model just stopped mid-sentence for no reason. The
    # finish_reason "length" is upstream's signal that decoding hit the
    # cap. "stop"/"tool_calls" are normal completions; we only warn on
    # "length" (truncation).
    if final_finish_reason == "length":
        emit("error", {"message": (
            f"[response truncated at QWEN_UI_MAX_TOKENS={CHAT_MAX_TOKENS} — "
            "the model hit its per-turn output cap. Bump it with "
            "QWEN_UI_MAX_TOKENS in the env, or ask the model to continue "
            "in the next turn.]"
        ), "kind": "warn"})
        logger.warning("response truncated at max_tokens=%d (finish_reason=length)",
                       CHAT_MAX_TOKENS)

    asst_msg: dict = {"role": "assistant"}
    asst_msg["content"] = visible_text or None
    if tool_calls:
        asst_msg["tool_calls"] = tool_calls
    return asst_msg, tool_calls


# Per-session cached dispatchers — keyed by session_id so two browser tabs
# don't pollute each other's caches. We use a WeakValueDictionary-like cap
# to bound growth; sessions older than DISPATCHER_CACHE_MAX evict LRU.
_CHAT_DISPATCHERS: dict[str, "object"] = {}
_CHAT_DISPATCHERS_LRU: list[str] = []
_CHAT_DISPATCHERS_LOCK = threading.Lock()
_DISPATCHER_CACHE_MAX = int(os.environ.get("QWEN_UI_DISPATCHER_CACHE", "32"))


def _get_chat_dispatcher(session_id: str):
    """Lazy-load and memoize one CachedDispatcher per chat session."""
    from agent_tools import CachedDispatcher  # lazy
    with _CHAT_DISPATCHERS_LOCK:
        d = _CHAT_DISPATCHERS.get(session_id)
        if d is None:
            d = CachedDispatcher()
            _CHAT_DISPATCHERS[session_id] = d
            _CHAT_DISPATCHERS_LRU.append(session_id)
            # LRU evict
            while len(_CHAT_DISPATCHERS_LRU) > _DISPATCHER_CACHE_MAX:
                evict = _CHAT_DISPATCHERS_LRU.pop(0)
                _CHAT_DISPATCHERS.pop(evict, None)
        else:
            # touch
            try:
                _CHAT_DISPATCHERS_LRU.remove(session_id)
            except ValueError:
                pass
            _CHAT_DISPATCHERS_LRU.append(session_id)
        return d


def _parse_tool_call(tc: dict) -> tuple[str, dict]:
    """Normalize one OpenAI-shape tool_call into (name, args_dict)."""
    fn = (tc.get("function") or {})
    name = fn.get("name") or "<unknown>"
    raw_args = fn.get("arguments")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {}
    return name, args


def _dispatch_tool_calls_batch(tool_calls: list[dict], session_id: str,
                                emit, task: str = "") -> tuple[list[dict], int, int]:
    """Dispatch all tool_calls for one assistant turn.

    Distinct calls run in PARALLEL (matching agent.py's CLI behavior).
    Identical (fn, args) pairs dedup to one dispatch with shared result.
    Cross-turn cache pulls repeat reads from the per-session CachedDispatcher.

    `task` is the most recent user message — used by triage_tool_result to
    score result relevance and prune off-topic noise (no-op when
    QWEN_TRIAGE_ENABLE != "1").

    Emits `tool_call` events up front (so the UI shows what's running) and
    one `tool_result` per input call. Returns
    (tool messages, wasted-result count, total calls) so the chat loop can
    detect repetition and break the model out. A result is "wasted" if it
    served no new information: byte-identical cache hit, semantic-near-dup
    notice from web_search, or a [REFUSED] marker from the URL guard.
    Byte-identical cache hits also surface in the per-call `cached` flag
    on tool_result events for the UI; the broader wasted-count signal is
    purely internal to the loop counter.
    """
    if not tool_calls:
        return [], 0, 0

    parsed = [(tc, *_parse_tool_call(tc)) for tc in tool_calls]

    # Up-front announcements so all spinners appear before parallel work.
    for tc, name, args in parsed:
        emit("tool_call", {"id": tc.get("id"), "name": name, "args": args})

    cdisp = _get_chat_dispatcher(session_id)
    t0 = time.monotonic()
    results = cdisp.dispatch_batch([(name, args) for _tc, name, args in parsed])
    duration_total_ms = int((time.monotonic() - t0) * 1000)

    from agent_tools import triage_tool_result  # lazy — avoid import-cycle risk

    out_msgs: list[dict] = []
    wasted_count = 0
    for (tc, name, _args), (raw, was_cached) in zip(parsed, results):
        result = raw if isinstance(raw, str) else str(raw)
        ok = not (result.startswith("[tool error]")
                  or result.startswith("[error]")
                  or result.startswith("[refused]"))
        truncated = False
        if len(result) > TOOL_RESULT_TRUNC:
            result = result[:TOOL_RESULT_TRUNC] + f"\n…[truncated {len(result) - TOOL_RESULT_TRUNC} chars]"
            truncated = True
        # Triage internally tries condense first (chunk-rank long results),
        # falls back to bge-small bi-encoder relevance scoring. The
        # `verdict` field tells which path fired: "condensed", "kept",
        # "low_relevance", or "skipped".
        result, triage_info = triage_tool_result(task, name, result)
        cdisp.record_reduction(triage_info)
        # "Wasted" classification — broader than `was_cached` so the loop
        # counter catches semantic near-duplicates (web_search dedup memo
        # at cosine ≥ 0.97 or high lexical overlap) and URL-guard refusals,
        # not just byte-identical arg matches. Without this, the model can
        # fire many near-duplicate web_searches in a row and the all-cached
        # threshold never trips.
        stripped = result.lstrip()
        is_wasted = (
            was_cached
            or stripped.startswith("[near-duplicate of earlier search")
            or stripped.startswith("[REFUSED")
        )
        if is_wasted:
            wasted_count += 1
        emit("tool_result", {
            "id": tc.get("id"),
            "name": name,
            "result": result,
            "ok": ok,
            "error": None if ok else result.split("\n", 1)[0][:200],
            "cached": bool(was_cached),
            "truncated": truncated,
            "triage": triage_info,
            # Per-call ms isn't separable when batched in parallel — report
            # the whole-batch wall time, which is the user-visible latency.
            "duration_ms": duration_total_ms,
        })
        out_msgs.append({
            "role": "tool",
            "tool_call_id": tc.get("id"),
            "content": result,
        })
    return out_msgs, wasted_count, len(parsed)


def _maybe_compact_chat(messages: list[dict], emit,
                          threshold: int | None = None,
                          preserve_latest_user: "bool | str" = "auto") -> list[dict]:
    """Mirror agent.py's auto-compaction in the chat tab.

    `threshold` overrides CONTEXT_WINDOW_TOKENS — callers pass a hard
    ceiling (e.g. 150k) for the pre-response safety check, leaving the
    post-response call on the soft (60k) default.

    `preserve_latest_user` modes:
      - True   — always keep the latest user message verbatim. Used for
                 pre-response compaction (model hasn't answered yet, so
                 we MUST keep the live question intact).
      - False  — always fold everything into the summary, including the
                 latest user message. Use sparingly.
      - "auto" — DEFAULT. Decide per-size: short user messages stay
                 verbatim (they're useful conversational context for the
                 next turn); large messages (file uploads, big pastes
                 over QWEN_UI_LARGE_USER_MSG_TOKENS, default 8000) fold
                 into the summary so they don't ride through unchanged
                 and produce a result barely smaller than the input.

    Uses the REAL Qwen tokenizer (`_count_chat_tokens`) when available so
    the trigger matches what the UI status bar reports."""
    try:
        from agent_tools import maybe_compact, approx_tokens
    except Exception:  # noqa: BLE001
        return messages
    # Prefer the real tokenizer used by the UI status bar — agreement
    # between "the bar says we're over" and "compact actually fires" is
    # what makes auto-compact trustworthy. Falls back to the char/4
    # estimate if the tokenizer isn't loaded.
    tokens_before = None
    try:
        tokens_before = _count_chat_tokens(messages, None)
    except Exception:  # noqa: BLE001
        tokens_before = None
    if tokens_before is None:
        try:
            tokens_before = approx_tokens(messages)
        except Exception:  # noqa: BLE001
            return messages
    effective_threshold = threshold if threshold is not None else CONTEXT_WINDOW_TOKENS
    if tokens_before < effective_threshold:
        return messages
    # Emit a heads-up BEFORE the slow compaction call so the client
    # knows the chat is intentionally paused, not hung. Compaction
    # itself is one full upstream call (no streaming) and on this
    # hardware it can take 1-3 minutes for 60k+ token histories.
    pre_note = (f"[auto-compacting context — {tokens_before // 1000}k tokens "
                f"of history exceeds the {effective_threshold // 1000}k "
                "threshold; one upstream summarization call follows…]")
    logger.info(pre_note)
    emit("error", {"message": pre_note, "kind": "info"})
    # Dedicated compaction event for the UI to render a "Compacting..."
    # indicator. Separate from the info message so the UI can show a
    # spinner / overlay rather than a one-line log entry.
    emit("compaction", {"status": "started",
                          "tokens_before": int(tokens_before),
                          "threshold": int(effective_threshold)})
    try:
        compacted = maybe_compact(messages, threshold=0)  # force when over
    except Exception as e:  # noqa: BLE001
        logger.warning("chat auto-compact failed: %s", e)
        emit("error", {"message": f"auto-compact failed: {e}", "kind": "error"})
        emit("compaction", {"status": "failed", "error": str(e)})
        return messages
    if not compacted:
        emit("error", {"message": "auto-compact returned no summary; "
                                  "continuing without compaction.", "kind": "error"})
        emit("compaction", {"status": "failed",
                             "error": "no summary produced"})
        return messages
    # Resolve "auto" → True (preserve) or False (fold) based on the size
    # of the latest user message. Short messages are conversational
    # context worth keeping verbatim; large ones (file uploads) need to
    # be summarized or compaction can't shrink the result meaningfully.
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if preserve_latest_user == "auto":
        large_thresh = int(os.environ.get("QWEN_UI_LARGE_USER_MSG_TOKENS",
                                            "8000"))
        if last_user_idx is None:
            preserve_latest_user = False
        else:
            # Use `message_content_tokens` (tokenizes the content directly)
            # rather than `_count_chat_tokens([msg])` which renders through
            # apply_chat_template and returns ~1 token for one-message
            # conversations regardless of size. The latter caused 70k-token
            # file uploads to be misclassified as "small".
            try:
                from agent_tools import message_content_tokens
                latest_user_tokens = message_content_tokens(
                    messages[last_user_idx])
            except Exception:  # noqa: BLE001
                # Last-ditch char-based fallback.
                content = messages[last_user_idx].get("content") or ""
                if isinstance(content, str):
                    latest_user_tokens = len(content) // 4
                else:
                    latest_user_tokens = len(str(content)) // 4
            preserve_latest_user = (latest_user_tokens < large_thresh)
            logger.info("compact: latest user message is %d tokens; "
                        "preserve_latest_user=%s (threshold=%d)",
                        latest_user_tokens, preserve_latest_user, large_thresh)
    # `maybe_compact` always preserves messages[:2] = [system, first_user]
    # in its head. If the FIRST user message is a large file upload (most
    # common pattern for "uploaded a doc, asked questions about it"), that
    # 70k-token message rides through compaction unchanged and the result
    # barely shrinks. Apply the same auto-mode rule: drop first_user from
    # the head when it's too big — the summary already captures it via
    # the @question section of the AGFMT compaction template.
    try:
        from agent_tools import message_content_tokens
        first_user_tokens = (message_content_tokens(compacted[1])
                             if len(compacted) >= 2
                             and compacted[1].get("role") == "user"
                             else 0)
    except Exception:  # noqa: BLE001
        first_user_tokens = 0
    large_thresh_head = int(os.environ.get("QWEN_UI_LARGE_USER_MSG_TOKENS",
                                            "8000"))
    if first_user_tokens >= large_thresh_head:
        logger.info("compact: first_user message is %d tokens (>= %d threshold) — "
                    "dropping from head; summary will cover its content",
                    first_user_tokens, large_thresh_head)
        # Reshape: drop compacted[1] (the huge first_user). New head is
        # just [system]; the summary stays as compacted[-1].
        compacted = [compacted[0]] + compacted[2:] if len(compacted) >= 2 else compacted
    if preserve_latest_user and last_user_idx is not None and last_user_idx > 1:
        # Keep the latest user message + its assistant response verbatim
        # AFTER the summary. Used when:
        #   - we're pre-response (model hasn't answered the live question)
        #   - the latest user message is short conversational context
        synthetic_summary = compacted[-1]
        sc = synthetic_summary.get("content") or ""
        cut = sc.find("[Now continue from this state")
        if cut > 0:
            sc = sc[:cut].rstrip()
        new_messages = (
            list(compacted[:-1])  # [system, (first_user)]
            + [{"role": "assistant", "content": sc}]
            + list(messages[last_user_idx:])  # latest user + tail
        )
    else:
        # Fold everything into the summary. Used when:
        #   - explicit preserve_latest_user=False
        #   - "auto" detected the latest user message is large (file upload)
        #   - there's no distinct latest user message (last_user_idx <= 1)
        new_messages = list(compacted)
    # Use the real tokenizer for the post-count too so the user sees
    # consistent numbers in the UI ("68k → 4k" not "50k → 3k").
    tokens_after = None
    try:
        tokens_after = _count_chat_tokens(new_messages, None)
    except Exception:  # noqa: BLE001
        tokens_after = None
    if tokens_after is None:
        try:
            tokens_after = approx_tokens(new_messages)
        except Exception:  # noqa: BLE001
            tokens_after = 0
    # Anti-retry guard: if compaction barely shrank (<10%), warn loudly
    # so the user knows the next turn won't help by re-running it. We
    # still apply the result (better than nothing), but the warning
    # surfaces a real bug instead of silently churning every turn.
    shrink_ratio = (tokens_before - tokens_after) / max(tokens_before, 1)
    if shrink_ratio < 0.10:
        warn = (f"[auto-compact ran but only shrank "
                f"{tokens_before // 1000}k → {tokens_after // 1000}k tokens "
                f"({shrink_ratio*100:.0f}%). Likely cause: latest user "
                f"message contains a large file/paste OR the summarization "
                f"call returned uncompressed text. Next turn will retry — "
                f"if this keeps happening, summarize the file content "
                f"yourself or start a fresh chat.]")
        logger.warning(warn)
        emit("error", {"message": warn, "kind": "error"})
    note = (f"[context auto-compacted: {tokens_before // 1000}k → "
            f"{tokens_after // 1000}k tokens at the {effective_threshold // 1000}k threshold]")
    logger.info(note)
    emit("error", {"message": note, "kind": "info"})
    emit("compaction", {"status": "done",
                          "tokens_before": int(tokens_before),
                          "tokens_after": int(tokens_after),
                          "threshold": int(effective_threshold)})
    return new_messages


def _run_chat_loop(messages: list[dict], emit, is_alive=lambda: True,
                   session_id: str = "") -> list[dict]:
    """Drive the chat: stream completion, dispatch tool_calls, repeat.
    Returns the full updated message list. Caller persists to session."""
    from agent_tools import _filtered_tools  # lazy import

    # Use chat-tab terse tool descriptions — saves ~5k baseline tokens vs
    # the agent.py defaults. Parameters/schema are preserved verbatim so
    # the model still calls each tool correctly.
    tools = _terse_tools(_chat_tool_tier(_filtered_tools()))

    # _run_chat_loop is invoked once per user message, so this is the right
    # place to reset per-task budgets (web_search/web_fetch caps, URL-guard
    # refusal counter). Without this, a long session would slowly starve
    # the model of search budget — by turn 5 the model gets `[REFUSED —
    # web_search cap of 8]` on every search.
    cdisp = _get_chat_dispatcher(session_id)
    cdisp.start_new_task()

    # Loop-detection: count consecutive turns where every tool call this turn
    # produced a "wasted" result — byte-identical cache hit, semantic-near-dup
    # notice from web_search (cosine ≥ 0.97 or high lexical overlap), or
    # [REFUSED] from the URL guard.
    # After CHAT_LOOP_BREAK_THRESHOLD such turns the model is provably stuck
    # retrying — inject a hard "synthesize and stop" nudge. Broader than
    # cache-hits-only because semantic near-duplicates aren't byte-identical
    # and so they're invisible to the dispatcher's args-keyed cache.
    consecutive_all_wasted = 0
    # Cumulative tool-call count for the runaway-cap. Set negative once
    # we've fired the cap so we don't re-inject on subsequent turns.
    total_tool_calls = 0
    # Loop-guard surfacing: if the proxy aborts a turn with the
    # `[loop-guard:]` marker, inject a course-correction nudge so the
    # next turn doesn't resume the loop. Mirrors agent.py (Round 6/15)
    # and agent_graph.py (Round 16). Single-fire per chat turn-loop.
    loop_guard_nudged = False

    # Hard ceiling = compact-before-response only when context is so large
    # that the upstream call would likely time out / OOM. Default 150k —
    # well above the soft 60k threshold but below the model's effective
    # 256k window. Can be tuned via QWEN_UI_HARD_COMPACT_AT.
    HARD_COMPACT_AT = int(
        os.environ.get("QWEN_UI_HARD_COMPACT_AT",
                       str(max(CONTEXT_WINDOW_TOKENS * 2, 150_000))))

    for turn in range(CHAT_MAX_TURNS):
        if not is_alive():
            return messages
        # Pre-response: only force-compact if context is dangerously high
        # (would likely time out). preserve_latest_user=True because the
        # model hasn't answered the live question yet — folding it into
        # a summary would lose what the model needs to answer.
        messages = _maybe_compact_chat(messages, emit,
                                         threshold=HARD_COMPACT_AT,
                                         preserve_latest_user=True)
        # Thinking is expensive (200-1000 tokens) and is rarely worth it on
        # tool-follow-ups, very-short conversational replies, or pure
        # imperative tool dispatch. We skip in those cases — same TPS-saver
        # the agent CLI uses, with two extra cases that real chat sessions
        # benefit from a lot:
        #   (1) trivial conversational chitchat like "thanks" / "ok"
        #   (2) clear single-tool imperatives like "call now()" or
        #       "fetch https://...". The model already routinely emits
        #       these decisions instantly without `<think>`.
        last_role = messages[-1].get("role") if messages else None
        enable_thinking = last_role != "tool"
        if enable_thinking and last_role == "user":
            last_user = (messages[-1].get("content") or "").strip()
            if _is_trivial_user_msg(last_user):
                enable_thinking = False

        try:
            asst_msg, tool_calls = _stream_one_completion(
                messages, tools, enable_thinking, emit, is_alive=is_alive
            )
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                err_body = ""
            # 502 from the proxy almost always means the worker behind it
            # is hung. Surface a clear remediation rather than the raw body.
            if e.code == 502:
                msg = ("inference worker hung — the model process stopped "
                       "responding. Run `qwen restart` in a terminal to bring "
                       "it back, then try again.")
            else:
                msg = f"upstream HTTP {e.code}: {err_body[:400]}"
            emit("error", {"message": msg})
            return messages
        except urllib.error.URLError as e:
            emit("error", {"message": f"upstream unreachable: {e.reason}"})
            return messages
        except Exception as e:  # noqa: BLE001
            emit("error", {"message": f"{type(e).__name__}: {e}"})
            return messages

        messages.append(asst_msg)
        emit("turn_end", {"had_tools": bool(tool_calls)})

        # Loop-guard surfacing: if the proxy aborted this turn, inject
        # a course-correction nudge BEFORE deciding whether to dispatch
        # any partial tool_calls. The model can emit a complete tool_call
        # XML and THEN slip into a loop — without this nudge the next
        # turn (after the tool result returns) often resumes the loop
        # because nothing in context tells the model its previous
        # response was truncated. See loop_guard_marker.py for the
        # false-positive-resistant detector contract.
        from loop_guard_marker import (
            is_proxy_abort_marker, extract_reason, harness_nudge_message,
        )
        asst_content = asst_msg.get("content") or ""
        if not loop_guard_nudged and is_proxy_abort_marker(asst_content):
            loop_guard_nudged = True
            reason = extract_reason(asst_content)
            note = (f"[loop-guard fired in chat: {reason} — injecting "
                    "course-correction nudge]")
            logger.info(note)
            emit("error", {"message": note, "kind": "info"})
            messages.append(harness_nudge_message(reason))
            # Don't dispatch any partial tool_calls — the abort cut them
            # off mid-flight. Continue to the next turn so the model
            # sees the nudge.
            continue

        if not tool_calls:
            # Post-response compaction (deferred soft-threshold check). The
            # user already got their answer; if context crossed 60k during
            # this turn, compact NOW so the next prompt benefits — without
            # making the user wait 1-3 minutes between sending a prompt
            # and seeing the response.
            try:
                # Default "auto" mode: short user messages stay verbatim,
                # large ones (files / big pastes over the threshold) fold
                # into the summary. Pre-response compaction at the hard
                # ceiling explicitly passes True to preserve the live
                # question.
                new_msgs = _maybe_compact_chat(messages, emit)
                if new_msgs is not messages:
                    messages[:] = new_msgs
            except Exception:  # noqa: BLE001
                pass
            return messages

        # Dispatch in batch: cross-turn cache + intra-turn dedup + parallel
        # execution of distinct calls. Mirrors agent.py CLI behavior.
        last_user = next(
            (m.get("content", "") for m in reversed(messages)
             if m.get("role") == "user" and isinstance(m.get("content"), str)),
            "",
        )
        # Seed the URL guard from the latest user message so URLs the user
        # pasted in are pre-authorized for web_fetch. Idempotent — a set.
        cdisp = _get_chat_dispatcher(session_id)
        if last_user:
            cdisp.note_text(last_user)
        tool_msgs, wasted_count, n_calls = _dispatch_tool_calls_batch(
            tool_calls, session_id, emit, task=last_user)
        messages.extend(tool_msgs)
        total_tool_calls += n_calls

        # Hard cap on total tool calls — fires before the cache-hit detector
        # for runaways that issue genuinely new queries each turn (e.g. a
        # yes/no question where the model fans out to 50+ web_searches).
        if total_tool_calls >= CHAT_MAX_TOOL_CALLS:
            note = (f"[tool-call cap: hit {total_tool_calls} total tool calls "
                    f"(>= {CHAT_MAX_TOOL_CALLS}) — injecting synthesize-now nudge]")
            logger.info(note)
            emit("error", {"message": note, "kind": "info"})
            messages.append({
                "role": "user",
                "content": (
                    f"[TOOL-CALL CAP: you've issued {total_tool_calls} tool "
                    f"calls across this conversation, which exceeds the cap "
                    f"of {CHAT_MAX_TOOL_CALLS}. Stop investigating. Synthesize "
                    "the best answer you can from the evidence above and "
                    "return it as plain text with NO further tool calls. If "
                    "you're genuinely missing critical info, say so and stop.]"
                ),
            })
            # Single shot: don't keep injecting on the next iteration.
            total_tool_calls = -10**9

        # Loop guard. If every tool call this turn was "wasted" — byte-cached,
        # a near-duplicate-search notice, or a [REFUSED] from the URL guard —
        # the model is asking the same questions over again or hammering an
        # exhausted budget. After 3 such turns we inject a hard "synthesize
        # and stop" message — same threshold as the CLI agent (agent.py).
        if n_calls > 0 and wasted_count == n_calls:
            consecutive_all_wasted += 1
        else:
            consecutive_all_wasted = 0
        if consecutive_all_wasted >= CHAT_LOOP_BREAK_THRESHOLD:
            note = (f"[loop-break: {consecutive_all_wasted} consecutive "
                    "all-wasted turns — injecting synthesize-now nudge]")
            logger.info(note)
            emit("error", {"message": note, "kind": "info"})
            messages.append({
                "role": "user",
                "content": (
                    f"[LOOP DETECTED: your last {consecutive_all_wasted} turns "
                    "issued tool calls that returned NO new information — every "
                    "result was either a cache hit, a near-duplicate-search "
                    "notice, or a [REFUSED] from an exhausted budget. Stop "
                    "investigating. Synthesize the best answer you can from the "
                    "evidence already gathered above and return it as plain text "
                    "with NO further tool calls. If genuinely missing critical "
                    "information, say so explicitly and stop.]"
                ),
            })
            consecutive_all_wasted = 0  # reset so we don't keep re-injecting

        # Re-anchor the client's context bar with the real prompt size
        # the NEXT loop iteration will pay. After a web_fetch returns 100k
        # chars this jump is large; this is exactly when the bar matters.
        per_turn_count = _count_chat_tokens(messages, tools)
        if per_turn_count is not None:
            emit("usage", {
                "prompt_tokens": per_turn_count,
                "completion_tokens": 0,
                "total_tokens": per_turn_count,
                "source": "interturn",
            })

    emit("error", {"message": f"hit max-turns cap ({CHAT_MAX_TURNS}) — stopping"})
    # Post-response compaction here too — even at the cap, the user saw
    # partial output and a fresh prompt benefits from compacted state.
    # "auto" decides whether to preserve the latest user message based
    # on its size (small=verbatim, large/file=fold into summary).
    try:
        new_msgs = _maybe_compact_chat(messages, emit)
        if new_msgs is not messages:
            messages[:] = new_msgs
    except Exception:  # noqa: BLE001
        pass
    return messages


# --------------------------------------------------------------------------
# request handler
# --------------------------------------------------------------------------

class UIHandler(BaseHTTPRequestHandler):
    server_version = "qwen-ui/0.1"

    def log_message(self, fmt, *args):  # noqa: A003
        logger.info("%s - %s", self.address_string(), fmt % args)

    # Wrap each verb in a top-level try so a programming bug in any individual
    # endpoint can't take down the connection silently. Without this, the
    # browser sees a bare ECONNRESET and the chat UI tab freezes its spinner
    # without telling the user why. With this, the user gets a JSON 500 they
    # can act on (or surface in toast).
    def _safe_dispatch(self, fn):
        try:
            return fn()
        except (BrokenPipeError, ConnectionResetError):
            return  # client gave up; nothing to send
        except Exception as e:  # noqa: BLE001
            logger.exception("unhandled error in %s: %s",
                             getattr(self, "path", "?"), e)
            try:
                _send_json(self, 500, {
                    "error": f"server error: {type(e).__name__}: {e}",
                    "path": getattr(self, "path", "?"),
                })
            except Exception:  # noqa: BLE001
                pass

    # ----- routing ----------------------------------------------------------

    def do_GET(self):  # noqa: N802
        return self._safe_dispatch(self._do_GET_inner)

    def do_DELETE(self):  # noqa: N802
        return self._safe_dispatch(self._do_DELETE_inner)

    def do_POST(self):  # noqa: N802
        return self._safe_dispatch(self._do_POST_inner)

    def do_PATCH(self):  # noqa: N802
        return self._safe_dispatch(self._do_PATCH_inner)

    def _do_GET_inner(self):
        path = self.path.split("?", 1)[0]

        if path == "/" or path == "/index.html":
            target = _safe_static_path("index.html")
            if target is None:
                _send_text(self, 200, _BOOTSTRAP_HTML, "text/html; charset=utf-8")
                return
            _serve_static(self, "index.html")
            return

        if path.startswith("/static/"):
            _serve_static(self, path[len("/static/"):])
            return

        if path == "/api/health":
            info = upstream_health()
            info["ui"] = "up"
            info["model_alias"] = MODEL_ALIAS
            info["model_id"] = _resolved_model_id  # may be None until first chat
            info["home"] = str(UI_HOME)
            info["queue"] = DISPATCHER.queue_depth()
            info["context_tokens"] = CONTEXT_WINDOW_TOKENS
            info["chat_max_tokens"] = CHAT_MAX_TOKENS
            _send_json(self, 200, info)
            return

        if path == "/api/health/extended":
            info = upstream_health()
            info["ui"] = "up"
            info["queue"] = DISPATCHER.queue_depth()
            info["model_id"] = _resolved_model_id
            info["chat_dispatchers"] = {
                "size": len(_CHAT_DISPATCHERS),
                "max": _DISPATCHER_CACHE_MAX,
                "session_ids": list(_CHAT_DISPATCHERS_LRU)[-10:],
            }
            # Per-dispatcher cache stats for the most recent N sessions
            cache_stats: list[dict] = []
            with _CHAT_DISPATCHERS_LOCK:
                for sid in list(_CHAT_DISPATCHERS_LRU)[-5:]:
                    d = _CHAT_DISPATCHERS.get(sid)
                    if d is None:
                        continue
                    try:
                        cache_stats.append({"session": sid, **d.stats()})
                    except Exception:  # noqa: BLE001
                        pass
            info["dispatcher_caches"] = cache_stats
            # Inference semaphore: count of "permits in use" by sampling
            # acquire() with timeout=0 (returns False if locked).
            try:
                _ensure_scripts_on_path()
                from agent_graph import _INFERENCE_SEM, _model_id  # type: ignore
                permit = _INFERENCE_SEM.acquire(blocking=False)
                if permit:
                    _INFERENCE_SEM.release()
                info["inference_lock"] = "free" if permit else "held"
            except Exception:  # noqa: BLE001
                info["inference_lock"] = "unknown"
            # Token budget visibility: surface the static cost the model
            # pays on every chat turn (system prompt + tool schemas).
            try:
                from agent_tools import _filtered_tools, real_tokens  # type: ignore
                tools_terse = _terse_tools(_chat_tool_tier(_filtered_tools()))
                # Real tokenizer for the baseline display. Empty conversation
                # means "system prompt only" — same as the per-turn fixed cost.
                sysprompt_text = _chat_system_prompt(os.getcwd())
                sysprompt_only = [{"role": "system", "content": sysprompt_text}]
                sysprompt_tokens = real_tokens(sysprompt_only, None)
                # Total = sysprompt + tools combined; tools only = subtract.
                total_tokens = real_tokens(sysprompt_only, tools_terse)
                tools_tokens = max(0, total_tokens - sysprompt_tokens)
                info["chat_baseline"] = {
                    "sysprompt_tokens": sysprompt_tokens,
                    "tools_tokens": tools_tokens,
                    "total_baseline_tokens": total_tokens,
                    "ultra_terse": _ULTRA_TERSE,
                    "rare_tools_enabled": _RARE_TOOLS_ENABLED,
                    "n_tools_in_chat": len(tools_terse),
                    "n_tools_total": len(_filtered_tools()),
                }
            except Exception:  # noqa: BLE001
                pass
            info["optimization_metrics"] = _optimization_metrics()
            _send_json(self, 200, info)
            return

        if path == "/api/file":
            qs = urllib.parse.parse_qs(self.path.partition("?")[2])
            raw = (qs.get("path") or [""])[0]
            _serve_local_file(self, raw)
            return

        if path == "/api/sessions":
            _send_json(self, 200, list_sessions())
            return

        m = re.fullmatch(r"/api/sessions/([\w\-]+)", path)
        if m:
            data = load_session(m.group(1))
            if data is None:
                _send_json(self, 404, {"error": "no such session"})
            else:
                _send_json(self, 200, data)
            return

        # ---- agents ----
        if path == "/api/agents":
            _send_json(self, 200, list_agents())
            return

        m = re.fullmatch(r"/api/agents/([\w\-]+)", path)
        if m:
            data = get_agent(m.group(1))
            if data is None:
                _send_json(self, 404, {"error": "no such agent"})
            else:
                _send_json(self, 200, data)
            return

        m = re.fullmatch(r"/api/agents/([\w\-]+)/runs", path)
        if m:
            _send_json(self, 200, _list_runs_local(m.group(1), limit=200))
            return

        m = re.fullmatch(r"/api/agents/([\w\-]+)/runs/([\w\-]+)", path)
        if m:
            data = get_run(m.group(1), m.group(2))
            if data is None:
                _send_json(self, 404, {"error": "no such run"})
            else:
                _send_json(self, 200, data)
            return

        if path == "/api/supervisor/status":
            sup = _supervisor_get("/status")
            _send_json(self, 200 if sup else 503, sup or {"error": "supervisor down"})
            return

        if path == "/api/graphs":
            _send_json(self, 200, _list_graphs())
            return

        # Graph runs archive
        if path == "/api/graph-runs":
            qs = urllib.parse.parse_qs(self.path.partition("?")[2])
            since = (qs.get("since") or [None])[0]
            graph_filter = (qs.get("graph") or [None])[0]
            try:
                limit = max(1, min(200, int((qs.get("limit") or ["50"])[0])))
            except ValueError:
                limit = 50
            _send_json(self, 200, list_graph_runs(
                since=since, limit=limit, graph_filter=graph_filter))
            return
        m = re.fullmatch(r"/api/graph-runs/([\w\-]+)", path)
        if m:
            data = get_graph_run(m.group(1))
            if data is None:
                _send_json(self, 404, {"error": "no such run"})
            else:
                _send_json(self, 200, data)
            return

        if path == "/api/mcps":
            _send_json(self, 200, _list_mcps())
            return

        _send_json(self, 404, {"error": "not found", "path": path})

    def _do_DELETE_inner(self):
        path = self.path.split("?", 1)[0]
        m = re.fullmatch(r"/api/sessions/([\w\-]+)", path)
        if m:
            ok = delete_session(m.group(1))
            _send_json(self, 200 if ok else 404, {"deleted": ok})
            return
        m = re.fullmatch(r"/api/graph-runs/([\w\-]+)", path)
        if m:
            ok = delete_graph_run(m.group(1))
            _send_json(self, 200 if ok else 404, {"deleted": ok})
            return
        m = re.fullmatch(r"/api/agents/([\w\-]+)", path)
        if m:
            status, body = delete_agent(m.group(1))
            _send_json(self, status, body)
            return
        _send_json(self, 404, {"error": "not found", "path": path})

    def _do_POST_inner(self):
        path = self.path.split("?", 1)[0]

        if path == "/api/chat/stream":
            self._handle_chat_stream()
            return

        if path == "/api/upload":
            self._handle_upload()
            return

        if path == "/api/graph/run":
            self._handle_graph_run()
            return

        if path == "/api/graph/stream":
            self._handle_graph_stream()
            return

        if path == "/api/graphs/create":
            self._handle_graph_create()
            return

        if path == "/api/graphs/preview":
            self._handle_graph_preview()
            return

        if path == "/api/graphs/save":
            self._handle_graph_save()
            return

        if path == "/api/graphs/delete":
            self._handle_graph_delete()
            return

        if path == "/api/mcps/register":
            self._handle_mcp_register()
            return

        if path == "/api/mcps/unregister":
            self._handle_mcp_unregister()
            return

        if path == "/api/agents":
            body = _read_json_body(self) or {}
            status, payload = create_agent(body)
            _send_json(self, status, payload)
            return

        m = re.fullmatch(r"/api/agents/([\w\-]+)/run-now", path)
        if m:
            sup = _supervisor_post(f"/agents/{m.group(1)}/run-now", {})
            if sup is None:
                _send_json(self, 503, {"error": "supervisor unreachable"})
            else:
                _send_json(self, 202, sup)
            return

        m = re.fullmatch(r"/api/agents/([\w\-]+)/cancel-run", path)
        if m:
            sup = _supervisor_post(f"/agents/{m.group(1)}/cancel-run", {})
            if sup is None:
                _send_json(self, 503, {"error": "supervisor unreachable"})
            else:
                _send_json(self, 200, sup)
            return

        # /api/sessions/<id>/extract — distill a chat into durable memories
        m = re.fullmatch(r"/api/sessions/([\w\-]+)/extract", path)
        if m:
            status, payload = extract_session_to_memory(m.group(1))
            _send_json(self, status, payload)
            return

        _send_json(self, 404, {"error": "not found", "path": path})

    def _do_PATCH_inner(self):
        path = self.path.split("?", 1)[0]
        m = re.fullmatch(r"/api/agents/([\w\-]+)", path)
        if m:
            body = _read_json_body(self) or {}
            status, payload = update_agent(m.group(1), body)
            _send_json(self, status, payload)
            return
        _send_json(self, 404, {"error": "not found", "path": path})

    # ----- agent graphs ----------------------------------------------------

    def _handle_graph_run(self) -> None:
        """Synchronously run a defined agent graph and return its outputs.

        Body: {"graph": "<name|path>", "inputs": {...}, "max_parallel": int}
        Returns: per-node outputs + per-node stats + total wall.
        """
        body = _read_json_body(self)
        if body is None:
            _send_json(self, 400, {"error": "expected JSON body"})
            return
        graph_arg = body.get("graph")
        inputs = body.get("inputs") or {}
        max_par = int(body.get("max_parallel", 4))
        if not graph_arg or not isinstance(inputs, dict):
            _send_json(self, 400, {"error": "graph (str) and inputs (object) required"})
            return
        try:
            g, gpath = _resolve_graph(graph_arg)
        except FileNotFoundError as e:
            _send_json(self, 404, {"error": str(e)})
            return
        except Exception as e:  # noqa: BLE001
            _send_json(self, 400, {"error": f"{type(e).__name__}: {e}"})
            return
        log: list[dict] = []
        def _cb(ev: dict) -> None:
            log.append(ev)
        graph_name = getattr(g, "name", os.path.basename(gpath))
        run_id = _new_graph_run_id()
        started_iso = _dt.datetime.now().isoformat(timespec="seconds")
        try:
            t0 = time.monotonic()
            out = g.run(inputs, verbose=False, max_parallel=max_par,
                         event_cb=_cb)
            wall = round(time.monotonic() - t0, 2)
        except Exception as e:  # noqa: BLE001
            # Archive the failed run so it shows up in the outputs panel.
            _persist_graph_run({
                "run_id": run_id, "graph": graph_name,
                "started_at": started_iso,
                "ended_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "wall_s": None, "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "inputs": inputs, "outputs": {}, "events": log,
            })
            _send_json(self, 500, {"error": f"{type(e).__name__}: {e}",
                                    "events": log, "run_id": run_id})
            return
        _persist_graph_run({
            "run_id": run_id, "graph": graph_name,
            "started_at": started_iso,
            "ended_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "wall_s": wall, "ok": True,
            "inputs": inputs, "outputs": out, "events": log,
        })
        _send_json(self, 200, {
            "graph": graph_name,
            "run_id": run_id,
            "wall_s": wall,
            "outputs": out,
            "events": log,
        })

    def _handle_graph_stream(self) -> None:
        """SSE-stream graph events (node_start, node_end, node_skipped) so a
        UI can show live progress per node without polling."""
        body = _read_json_body(self)
        if body is None:
            _send_json(self, 400, {"error": "expected JSON body"})
            return
        graph_arg = body.get("graph")
        inputs = body.get("inputs") or {}
        max_par = int(body.get("max_parallel", 4))
        try:
            g, gpath = _resolve_graph(graph_arg)
        except FileNotFoundError as e:
            _send_json(self, 404, {"error": str(e)})
            return
        except Exception as e:  # noqa: BLE001
            _send_json(self, 400, {"error": f"{type(e).__name__}: {e}"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        # Cross-thread queue: graph runs on a background thread, the request
        # thread pulls events off the queue and writes them as SSE frames.
        import queue as _q
        q: _q.Queue = _q.Queue()
        events_log: list[dict] = []
        graph_name = getattr(g, "name", os.path.basename(gpath))
        run_id = _new_graph_run_id()
        started_iso = _dt.datetime.now().isoformat(timespec="seconds")

        def _cb(ev: dict) -> None:
            events_log.append(ev)
            q.put(("event", ev))

        def _worker() -> None:
            t0 = time.monotonic()
            try:
                out = g.run(inputs, verbose=False, max_parallel=max_par,
                             event_cb=_cb)
                wall = round(time.monotonic() - t0, 2)
                _persist_graph_run({
                    "run_id": run_id, "graph": graph_name,
                    "started_at": started_iso,
                    "ended_at": _dt.datetime.now().isoformat(timespec="seconds"),
                    "wall_s": wall, "ok": True,
                    "inputs": inputs, "outputs": out, "events": events_log,
                })
                q.put(("done", {"wall_s": wall, "outputs": out,
                                "run_id": run_id}))
            except Exception as e:  # noqa: BLE001
                _persist_graph_run({
                    "run_id": run_id, "graph": graph_name,
                    "started_at": started_iso,
                    "ended_at": _dt.datetime.now().isoformat(timespec="seconds"),
                    "wall_s": None, "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "inputs": inputs, "outputs": {}, "events": events_log,
                })
                q.put(("error", {"message": f"{type(e).__name__}: {e}",
                                  "run_id": run_id}))

        threading.Thread(target=_worker, daemon=True).start()
        try:
            # Tell the client the run id up front so the UI can deep-link
            # to the archive entry even before completion.
            self.wfile.write(_sse_frame("run_id", {"run_id": run_id}))
            self.wfile.flush()
            while True:
                kind, payload = q.get()
                if kind == "event":
                    ev_kind = payload.get("kind", "event")
                    self.wfile.write(_sse_frame(ev_kind, payload))
                    self.wfile.flush()
                elif kind == "done":
                    self.wfile.write(_sse_frame("done", payload))
                    self.wfile.flush()
                    break
                elif kind == "error":
                    self.wfile.write(_sse_frame("error", payload))
                    self.wfile.flush()
                    break
        except (BrokenPipeError, ConnectionResetError):
            return

    def _handle_graph_create(self) -> None:
        """Run the graph_designer to turn a NL description into a saved graph.

        Body: {"description": "..."}.
        Returns: {"ok", "name", "path", "spec"} on success;
                  {"ok": false, "error": ...} on validation/load failure.
        """
        body = _read_json_body(self)
        if body is None:
            _send_json(self, 400, {"error": "expected JSON body"})
            return
        desc = (body.get("description") or "").strip()
        if len(desc) < 8:
            _send_json(self, 400, {"error": "description must be at least 8 chars"})
            return
        _ensure_scripts_on_path()
        try:
            from graph_designer import design_and_save  # type: ignore
        except Exception as e:  # noqa: BLE001
            _send_json(self, 500, {"error": f"cannot import graph_designer: {e}"})
            return
        # Same dispatcher discipline as /api/graphs/preview — see the
        # comment in _handle_graph_preview.
        DISPATCHER.acquire(PRIO_NORMAL, label="graph_create")
        try:
            result = design_and_save(desc)
        except Exception as e:  # noqa: BLE001
            _send_json(self, 500, {"error": f"{type(e).__name__}: {e}"})
            return
        finally:
            DISPATCHER.release()
        _send_json(self, 200 if result.get("ok") else 422, result)

    def _handle_graph_preview(self) -> None:
        """Two-phase graph creation step 1: design without saving.

        Body: {"description": "..."}
        Returns: {"ok", "spec", "code"} so the UI can render an editor before
        committing the file. The user can mutate `spec` (rename nodes,
        re-shuffle tools, edit goals) and POST back to /api/graphs/save.
        """
        body = _read_json_body(self)
        if body is None:
            _send_json(self, 400, {"error": "expected JSON body"})
            return
        desc = (body.get("description") or "").strip()
        if len(desc) < 8:
            _send_json(self, 400, {"error": "description must be at least 8 chars"})
            return
        _ensure_scripts_on_path()
        try:
            from graph_designer import design_preview  # type: ignore
        except Exception as e:  # noqa: BLE001
            _send_json(self, 500, {"error": f"cannot import graph_designer: {e}"})
            return
        # Graph design is a long single-shot upstream call (~30s for complex
        # specs). Hold the dispatcher at NORMAL priority so a concurrent
        # chat (HIGH) can jump ahead — without this, design bypasses the
        # UI dispatcher and chat queues behind it at the dflash layer.
        DISPATCHER.acquire(PRIO_NORMAL, label="graph_preview")
        try:
            result = design_preview(desc)
        except Exception as e:  # noqa: BLE001
            _send_json(self, 500, {"error": f"{type(e).__name__}: {e}"})
            return
        finally:
            DISPATCHER.release()
        _send_json(self, 200 if result.get("ok") else 422, result)

    def _handle_graph_save(self) -> None:
        """Two-phase graph creation step 2: persist a (possibly user-edited)
        spec.

        Body: {"spec": {...}, "description": "..."}
        Returns: {"ok", "name", "path"} on success.
        """
        body = _read_json_body(self)
        if body is None:
            _send_json(self, 400, {"error": "expected JSON body"})
            return
        spec = body.get("spec")
        if not isinstance(spec, dict):
            _send_json(self, 400, {"error": "spec (object) is required"})
            return
        description = (body.get("description") or "").strip()
        _ensure_scripts_on_path()
        try:
            from graph_designer import save_spec  # type: ignore
        except Exception as e:  # noqa: BLE001
            _send_json(self, 500, {"error": f"cannot import graph_designer: {e}"})
            return
        try:
            result = save_spec(spec, description=description)
        except Exception as e:  # noqa: BLE001
            _send_json(self, 500, {"error": f"{type(e).__name__}: {e}"})
            return
        _send_json(self, 200 if result.get("ok") else 422, result)

    def _handle_graph_delete(self) -> None:
        """Delete a saved graph from examples/. Body: {"name": "..."}.

        Path traversal is blocked: name must match `[a-z0-9_]+`.
        """
        body = _read_json_body(self)
        if body is None or not isinstance(body.get("name"), str):
            _send_json(self, 400, {"error": "name (string) is required"})
            return
        name = body["name"].strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,60}", name):
            _send_json(self, 400, {"error": "invalid graph name"})
            return
        d = _graphs_dir()
        path = os.path.join(d, f"{name}_graph.py")
        # Resolve and confirm it stays under the graphs dir
        try:
            real_d = os.path.realpath(d)
            real_p = os.path.realpath(path)
            if os.path.commonpath([real_d, real_p]) != real_d:
                _send_json(self, 400, {"error": "path escapes examples dir"})
                return
        except Exception:  # noqa: BLE001
            _send_json(self, 400, {"error": "invalid path"})
            return
        if not os.path.exists(path):
            _send_json(self, 404, {"error": "no such graph"})
            return
        try:
            os.remove(path)
        except OSError as e:
            _send_json(self, 500, {"error": f"delete failed: {e}"})
            return
        _send_json(self, 200, {"ok": True, "deleted": name})

    def _handle_mcp_register(self) -> None:
        """Register a custom MCP server in the persistent registry.
        Body: {"name": "...", "url": "http://...", "headers": {...}, "tools": [...]}.

        Tools is an optional list of preconfigured tool descriptors so the
        chat agent can call them as `mcp_<name>__<tool>`. URL must respond
        to a `POST /tools/<tool>` with JSON body of args. Minimal contract;
        sufficient for ad-hoc MCPs without a full JSON-RPC client.
        """
        body = _read_json_body(self)
        if body is None or not body.get("name") or not body.get("url"):
            _send_json(self, 400, {"error": "name and url are required"})
            return
        try:
            entry = _mcp_register(body)
        except ValueError as e:
            _send_json(self, 400, {"error": str(e)})
            return
        _send_json(self, 201, entry)

    def _handle_mcp_unregister(self) -> None:
        body = _read_json_body(self)
        if body is None or not body.get("name"):
            _send_json(self, 400, {"error": "name is required"})
            return
        ok = _mcp_unregister(body["name"])
        _send_json(self, 200 if ok else 404, {"ok": ok})

    # ----- file uploads (paperclip / drag-drop) ----------------------------

    def _handle_upload(self) -> None:
        """Multipart upload from the chat composer. Returns one JSON object
        per uploaded file with a stable id the client passes back as
        `attachments` in the next chat-stream POST."""
        files, err = _parse_multipart(self, UPLOAD_MAX_BYTES)
        if err:
            _send_json(self, 400, {"error": err})
            return
        if not files:
            _send_json(self, 400, {"error": "no files in upload"})
            return
        results: list[dict] = []
        had_error = False
        for f in files:
            status, payload = _save_upload(f)
            payload["status"] = status
            if status >= 400:
                had_error = True
            # Don't send the full extracted text back to the client — it's
            # already on disk and will be merged into the chat call. Trim
            # for the JSON response.
            payload.pop("text", None)
            results.append(payload)
        _send_json(self, 207 if had_error else 200, {"files": results})

    # ----- chat ------------------------------------------------------------

    def _maybe_handle_slash_command(self, cmd_line: str,
                                      client_messages: list[dict],
                                      session_id: str,
                                      cwd: str) -> bool:
        """Server-side slash-command router. Returns True if the command was
        handled (response streamed, SSE closed). Returns False if the input
        wasn't a recognized slash command — caller continues to the model.

        Each command emits its result through the same SSE channel the
        model uses so the UI renders it identically. The synthetic assistant
        reply is persisted to the session so reloading the page shows it.
        """
        # Tokenize the command. Special-case `/graph` (single slash, two
        # subcommands) and `/graph_compose` (underscore).
        line = cmd_line.lstrip()
        if not line.startswith("/"):
            return False
        head, _, rest = line[1:].partition(" ")
        rest = rest.strip()
        cmd = head.lower()

        # Reject unknown slash commands — pass through to model so the model
        # can interpret things like "/api/foo" the user pasted accidentally.
        if cmd not in {"help", "graphs", "graph", "graph_compose",
                        "scratchpad", "tools"}:
            return False

        # Open SSE response and stream the synthetic reply.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True

        # We hold the per-session lock for the synthetic reply just like a
        # model call would — prevents another tab from racing in.
        session_lock = _session_chat_lock(session_id)
        if not session_lock.acquire(blocking=False):
            self.wfile.write(_sse_frame("error", {
                "message": "another chat is in flight for this session",
            }))
            return True

        write_lock = threading.Lock()
        closed = {"v": False}

        def emit(event: str, data: dict) -> None:
            if closed["v"]:
                return
            try:
                with write_lock:
                    self.wfile.write(_sse_frame(event, data))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                closed["v"] = True

        def emit_text(text: str) -> None:
            # Stream as a `content` event with a `delta` field so the UI's
            # existing chat-renderer (which already handles model streaming)
            # treats this identically to a model response.
            for chunk in [text[i:i+512] for i in range(0, len(text), 512)]:
                emit("content", {"delta": chunk})

        try:
            emit("started", {
                "session_id": session_id,
                "queue": DISPATCHER.queue_depth(),
                "slash": cmd,
            })

            reply_text = ""
            if cmd == "help":
                reply_text = _SLASH_HELP

            elif cmd == "tools":
                from agent_tools import _filtered_tools as _ft  # type: ignore
                tools = _chat_tool_tier(_ft())
                lines = ["**Chat-tier tools** (in addition to anything the model calls):\n"]
                for t in tools:
                    fn = (t.get("function") or {})
                    name = fn.get("name", "?")
                    blurb = _CHAT_TOOL_BLURBS.get(name) or (
                        (fn.get("description") or "").split(". ", 1)[0][:120]
                    )
                    lines.append(f"- `{name}` — {blurb}")
                reply_text = "\n".join(lines)

            elif cmd == "graphs":
                info = _list_graphs()
                graphs = info.get("graphs") or []
                if not graphs:
                    reply_text = "_no graphs defined under `examples/`._"
                else:
                    lines = ["**Available graphs** (call `/graph <name> [json-inputs]` to run):\n"]
                    lines.append("| name | nodes | entry inputs |")
                    lines.append("|---|---|---|")
                    for g in graphs:
                        inputs = ", ".join(g.get("entry_inputs") or []) or "—"
                        lines.append(
                            f"| `{g.get('name','?')}` | {g.get('nodes','?')} | {inputs} |"
                        )
                    reply_text = "\n".join(lines)

            elif cmd == "graph":
                # `/graph <name> {json}` — run a graph and stream events.
                if not rest:
                    reply_text = (
                        "Usage: `/graph <name> [json-inputs]`. "
                        "Run `/graphs` to see available graphs."
                    )
                else:
                    name, _, inputs_raw = rest.partition(" ")
                    name = name.strip()
                    try:
                        inputs = (json.loads(inputs_raw.strip())
                                  if inputs_raw.strip() else {})
                    except json.JSONDecodeError as e:
                        reply_text = f"**Error**: inputs must be valid JSON ({e})."
                    else:
                        try:
                            g, gpath = _resolve_graph(name)
                        except FileNotFoundError as e:
                            reply_text = f"**Error**: {e}"
                        except Exception as e:  # noqa: BLE001
                            reply_text = (f"**Error**: failed to load graph "
                                          f"`{name}`: {type(e).__name__}: {e}")
                        else:
                            # Stream events inline. Each node_start /
                            # node_end shows up as a short status line in
                            # the chat so the user sees progress.
                            emit_text(
                                f"**Running graph `{name}`** with inputs "
                                f"`{json.dumps(inputs)}`…\n\n"
                            )
                            events_log: list[dict] = []
                            def _cb(ev: dict) -> None:
                                events_log.append(ev)
                                k = ev.get("kind", "?")
                                if k == "node_start":
                                    emit_text(f"- ▸ `{ev.get('node','?')}` starting\n")
                                elif k == "node_end":
                                    wall = ev.get("wall_s")
                                    nstr = (f" ({wall:.1f}s)"
                                             if isinstance(wall, (int, float))
                                             else "")
                                    emit_text(f"- ✓ `{ev.get('node','?')}` done{nstr}\n")
                                elif k == "node_skipped":
                                    emit_text(f"- ⊘ `{ev.get('node','?')}` skipped\n")
                            graph_name = getattr(g, "name", os.path.basename(gpath))
                            run_id = _new_graph_run_id()
                            started_iso = _dt.datetime.now().isoformat(timespec="seconds")
                            t0 = time.monotonic()
                            try:
                                out = g.run(inputs, verbose=False,
                                             max_parallel=4, event_cb=_cb)
                                wall = round(time.monotonic() - t0, 2)
                                _persist_graph_run({
                                    "run_id": run_id, "graph": graph_name,
                                    "started_at": started_iso,
                                    "ended_at": _dt.datetime.now().isoformat(timespec="seconds"),
                                    "wall_s": wall, "ok": True,
                                    "inputs": inputs, "outputs": out,
                                    "events": events_log,
                                })
                                # Render outputs as collapsed details blocks.
                                emit_text(f"\n**Done in {wall}s** "
                                          f"(run id `{run_id}`). Outputs:\n\n")
                                for node, outputs in (out or {}).items():
                                    if isinstance(outputs, dict) and outputs.get("_skipped"):
                                        continue
                                    emit_text(f"<details><summary><code>{node}</code></summary>\n\n")
                                    emit_text("```json\n" + json.dumps(outputs, indent=2, default=str) + "\n```\n\n")
                                    emit_text("</details>\n\n")
                                reply_text = ""  # already streamed
                            except Exception as e:  # noqa: BLE001
                                _persist_graph_run({
                                    "run_id": run_id, "graph": graph_name,
                                    "started_at": started_iso,
                                    "ended_at": _dt.datetime.now().isoformat(timespec="seconds"),
                                    "wall_s": None, "ok": False,
                                    "error": f"{type(e).__name__}: {e}",
                                    "inputs": inputs, "outputs": {},
                                    "events": events_log,
                                })
                                emit_text(f"\n**Error**: {type(e).__name__}: {e}\n")
                                reply_text = ""

            elif cmd == "graph_compose":
                if not rest or len(rest.strip()) < 8:
                    reply_text = ("Usage: `/graph_compose <description ≥8 chars>`. "
                                   "Provide a paragraph describing the pipeline.")
                else:
                    emit_text(f"**Designing graph from**:\n\n> {rest[:300]}…\n\n")
                    _ensure_scripts_on_path()
                    try:
                        from graph_designer import design_and_save  # type: ignore
                    except Exception as e:  # noqa: BLE001
                        reply_text = f"**Error**: graph_designer unavailable ({e})."
                    else:
                        DISPATCHER.acquire(PRIO_NORMAL, label="slash_graph_compose")
                        try:
                            result = design_and_save(rest)
                        except Exception as e:  # noqa: BLE001
                            reply_text = f"**Error**: {type(e).__name__}: {e}"
                            result = None
                        finally:
                            DISPATCHER.release()
                        if isinstance(result, dict) and result.get("ok"):
                            name = result.get("name", "?")
                            path = result.get("path", "?")
                            spec = result.get("spec") or {}
                            n_nodes = len(spec.get("nodes") or [])
                            reply_text = (
                                f"**Saved graph `{name}`** "
                                f"({n_nodes} nodes) at `{path}`.\n\n"
                                f"Run it with `/graph {name} {{...}}` or call "
                                f"`agent_graph_run` from a model turn."
                            )
                        elif result is not None:
                            reply_text = (
                                f"**Design rejected**: "
                                f"{result.get('error', 'unknown error')}"
                            )

            elif cmd == "scratchpad":
                # `/scratchpad [read|clear] [key]`. Default: read default pad.
                tokens = rest.split() if rest else []
                action = tokens[0].lower() if tokens else "read"
                key = tokens[1] if len(tokens) > 1 else "default"
                try:
                    from agent_tools import scratchpad as _sp
                    out = _sp(action=action, key=key, content="")
                except Exception as e:  # noqa: BLE001
                    out = f"[error] {type(e).__name__}: {e}"
                reply_text = "```\n" + out + "\n```"

            if reply_text:
                emit_text(reply_text)

            # Persist the synthetic exchange to the session so a reload
            # shows it identically to a model response.
            new_messages = list(client_messages)
            # The latest user msg is already in new_messages. Append the
            # assistant synthetic message AFTER it.
            synthetic_content = reply_text or "(slash command output streamed above)"
            new_messages.append({
                "role": "assistant",
                "content": synthetic_content,
            })
            try:
                _persist_session(session_id, new_messages)
            except Exception as e:  # noqa: BLE001
                logger.warning("session persist failed (slash): %s", e)

            emit("done", {
                "session_id": session_id,
                "messages": _strip_messages_for_client(new_messages),
            })
        except Exception as e:  # noqa: BLE001
            logger.exception("slash command crashed: %s", cmd)
            emit("error", {"message": f"slash error: {type(e).__name__}: {e}"})
        finally:
            try:
                session_lock.release()
            except (RuntimeError, NameError):
                pass
        return True

    def _handle_chat_stream(self) -> None:
        body = _read_json_body(self)
        if body is None:
            _send_json(self, 400, {"error": "invalid json"})
            return
        client_messages = body.get("messages") or []
        if not isinstance(client_messages, list) or not client_messages:
            _send_json(self, 400, {"error": "messages required"})
            return
        # Validate message shape early — reject malformed rows BEFORE we
        # spend tokens on the upstream call. The previous behavior was to
        # let the model see {"role": "EVIL"} or {"content": 42} and burn
        # ~30s generating gibberish.
        ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
        for i, m in enumerate(client_messages):
            if not isinstance(m, dict):
                _send_json(self, 400, {"error": f"messages[{i}] not an object"})
                return
            role = m.get("role")
            if role not in ALLOWED_ROLES:
                _send_json(self, 400,
                    {"error": f"messages[{i}].role must be one of "
                              f"{sorted(ALLOWED_ROLES)}, got {role!r}"})
                return
            content = m.get("content")
            # OpenAI tool-call messages can omit content; otherwise must be str
            # or a list (multimodal — accept and let upstream reject if shape
            # is wrong, since enumerating valid multimodal shapes here would
            # quickly drift from upstream).
            if content is not None and not isinstance(content, (str, list)):
                _send_json(self, 400,
                    {"error": f"messages[{i}].content must be string or null"})
                return
            # Cap individual message length so a single message can't OOM
            # the proxy/upstream. Default is 800k chars (~200k tokens of
            # dense English / ~280k of code) — well within Qwen3.6's 256k
            # context window. Previous 200k cap rejected file uploads that
            # were within UPLOAD_MAX_TEXT after the UI added framing prose.
            # Override with QWEN_UI_MAX_MSG_CHARS.
            max_msg_chars = int(os.environ.get("QWEN_UI_MAX_MSG_CHARS",
                                                "800000"))
            if isinstance(content, str) and len(content) > max_msg_chars:
                _send_json(self, 413,
                    {"error": f"messages[{i}].content too large "
                              f"({len(content)} > {max_msg_chars} chars). "
                              f"Set QWEN_UI_MAX_MSG_CHARS to raise the cap, "
                              f"or split the content across messages."})
                return
        if len(client_messages) > 500:
            _send_json(self, 413,
                {"error": f"too many messages ({len(client_messages)} > 500)"})
            return
        cwd = body.get("cwd") or os.getcwd()
        # Session IDs that hit the filesystem must be safe identifiers — no
        # path separators, no parent refs, bounded length. Default-generated
        # IDs match `YYYYMMDD-HHMMSS-xxxxxx`; we accept any [\w-]{1,80}.
        raw_sid = body.get("session_id")
        if raw_sid is None:
            session_id = _new_session_id()
        elif not isinstance(raw_sid, str) or not _SESSION_ID_RE.fullmatch(raw_sid):
            _send_json(self, 400, {"error": "session_id must match "
                                            "[a-zA-Z0-9_-]{1,80}"})
            return
        else:
            session_id = raw_sid

        # Inject system prompt if the client didn't provide one.
        if not (client_messages and client_messages[0].get("role") == "system"):
            client_messages = [
                {"role": "system", "content": _chat_system_prompt(cwd)},
                *client_messages,
            ]

        # Slash-command interception. If the latest user message is a slash
        # command (`/graphs`, `/graph <name>`, `/graph_compose <desc>`,
        # `/scratchpad ...`, `/help`), handle it server-side and short-
        # circuit the model call. The synthetic assistant reply is streamed
        # back via the same SSE channel so the UI renders it identically to
        # a model response. This is the chat<->graph integration: the user
        # can directly invoke graph operations from the chat box without
        # leaving the conversation for the graphs panel.
        last_user = None
        for m in reversed(client_messages):
            if m.get("role") == "user":
                last_user = m
                break
        if (last_user is not None
                and isinstance(last_user.get("content"), str)
                and last_user["content"].lstrip().startswith("/")
                and not body.get("skip_slash")):
            handled = self._maybe_handle_slash_command(
                last_user["content"].strip(),
                client_messages,
                session_id,
                cwd,
            )
            if handled:
                return

        # Merge file attachments into the LATEST user message (if any).
        # The client sends `attachments` as a list of upload IDs from
        # /api/upload — we read each file's stored text and prepend it to
        # the user's message wrapped in a <details>…</details> block, so
        # the UI's markdown renderer collapses each file by default.
        attachments = body.get("attachments") or []
        if attachments:
            try:
                client_messages = _merge_attachments_into_last_user(
                    list(client_messages), attachments
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("attachment merge failed: %s", e)

        # Per-session try-lock: refuse a second concurrent chat against the
        # same session_id rather than silently losing one tab's messages.
        # Browsers normally produce a fresh session_id per tab, so this only
        # fires if the user explicitly reuses an id (e.g., two tabs pinned
        # to the same session, or a misbehaving API client).
        session_lock = _session_chat_lock(session_id)
        if not session_lock.acquire(blocking=False):
            _send_json(self, 409, {
                "error": "another chat is already in flight for this "
                         "session_id; close the other client or pick a "
                         "different session_id"
            })
            return

        # Open SSE response. Once we send the headers we own the wire until
        # we close — any errors must come through `error` events. We send
        # `Connection: close` and set close_connection=True so the underlying
        # http.server actually drops the socket once we return; otherwise
        # keep-alive leaves curl/EventSource hanging waiting for more bytes.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True

        write_lock = threading.Lock()
        closed = {"v": False}

        def emit(event: str, data: dict) -> None:
            if closed["v"]:
                return
            try:
                with write_lock:
                    self.wfile.write(_sse_frame(event, data))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                closed["v"] = True

        try:
            emit("started", {
                "session_id": session_id,
                "queue": DISPATCHER.queue_depth(),
            })
            # Eagerly resolve the upstream model id BEFORE entering the
            # chat loop, so an early compaction call (which goes through
            # agent_tools._post_chat) sees the right env var and doesn't
            # 400 against the alias. Cheap if already cached.
            resolve_model_id()
            # Pre-flight token count using the real model tokenizer (since
            # dflash doesn't return `usage` over the stream). Anchors the
            # client's context bar BEFORE the first delta arrives.
            try:
                _toolset_for_count = None
                from agent_tools import _filtered_tools as _ft  # lazy
                _toolset_for_count = _terse_tools(_ft())
            except Exception:  # noqa: BLE001
                _toolset_for_count = None
            preflight_tokens = _count_chat_tokens(client_messages, _toolset_for_count)
            if preflight_tokens is not None:
                emit("usage", {
                    "prompt_tokens": preflight_tokens,
                    "completion_tokens": 0,
                    "total_tokens": preflight_tokens,
                    "source": "preflight",
                })
            messages = _run_chat_loop(
                list(client_messages),
                emit,
                is_alive=lambda: not closed["v"],
                session_id=session_id,
            )
            # Post-flight count: includes everything the conversation now
            # holds (assistant turns, tool results). This is what'll be
            # re-sent next turn, i.e. the actual context cost.
            postflight_tokens = _count_chat_tokens(messages, _toolset_for_count)
            if postflight_tokens is not None:
                emit("usage", {
                    "prompt_tokens": postflight_tokens,
                    "completion_tokens": 0,
                    "total_tokens": postflight_tokens,
                    "source": "postflight",
                })
            # Persist BEFORE emitting `done` so that any client request to
            # /api/sessions/<id> immediately after receiving `done` finds
            # an up-to-date sidecar (with the right updated_at). Otherwise
            # there's a small race where the client reloads the session
            # before the disk write finishes.
            try:
                _persist_session(session_id, messages)
            except Exception as e:  # noqa: BLE001
                logger.warning("session persist failed: %s", e)
            emit("done", {
                "session_id": session_id,
                "messages": _strip_messages_for_client(messages),
            })
        except Exception as e:  # noqa: BLE001
            logger.exception("chat stream crashed")
            emit("error", {"message": f"server error: {type(e).__name__}: {e}"})
        finally:
            # Always release the per-session lock, even if the stream
            # crashed or the client disconnected. Otherwise the session_id
            # would be permanently stuck at 409 until the UI restart.
            try:
                session_lock.release()
            except (RuntimeError, NameError):
                pass


# --------------------------------------------------------------------------
# session persistence (placeholder — fully wired in step 5)
# --------------------------------------------------------------------------

_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_\-]{1,80}")


def _new_session_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def _strip_messages_for_client(messages: list[dict]) -> list[dict]:
    """Drop the leading system message before sending to the browser — the
    client doesn't need it (it's regenerated each turn from the cwd) and
    leaking long system prompts wastes round-trip bytes."""
    if messages and messages[0].get("role") == "system":
        return messages[1:]
    return messages


# How many messages have been persisted for each in-memory session. Lets
# _persist_session append only the delta since last call instead of
# rewriting the entire file every turn. Keyed by session_id.
_PERSISTED_COUNTS: dict[str, int] = {}
_PERSIST_LOCK = threading.Lock()


# Per-session chat-stream lock. Two browser tabs writing to the same
# session_id used to silently lose one tab's messages because each chat
# only knew about its own client-supplied history; whichever persisted
# second would see n_persisted already covering the count and append
# nothing. Serialize chats per-session so the second one sees the first's
# completed turns in its replay history.
_SESSION_CHAT_LOCKS: dict[str, threading.Lock] = {}
_SESSION_CHAT_LOCKS_GUARD = threading.Lock()


def _session_chat_lock(session_id: str) -> threading.Lock:
    with _SESSION_CHAT_LOCKS_GUARD:
        lk = _SESSION_CHAT_LOCKS.get(session_id)
        if lk is None:
            lk = threading.Lock()
            _SESSION_CHAT_LOCKS[session_id] = lk
        return lk


def _idx_path(session_id: str) -> Path:
    """Sidecar metadata file (id, title, updated_at, n_messages). Cheap to
    read for the sidebar without parsing the main JSONL."""
    return SESSIONS_DIR / f"{session_id}.idx.json"


def _persist_session(session_id: str, messages: list[dict]) -> None:
    """Append-only persistence.

    Old design rewrote the whole conversation every turn (O(n²) writes
    over the lifetime of a session). New design appends only the messages
    that haven't been seen yet, and writes a small sidecar `.idx.json`
    file for the sidebar's list_sessions().

    Backward compat: existing single-line "wrapped" sessions are detected
    and migrated to the new format on first persist call for that session.
    """
    cleaned = _strip_messages_for_client(messages)
    path = SESSIONS_DIR / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = _idx_path(session_id)

    with _PERSIST_LOCK:
        # If we don't have a count for this session yet, see whether a file
        # is on disk. If it is, decide whether to keep it (new format) or
        # migrate it (old single-line format).
        n_persisted = _PERSISTED_COUNTS.get(session_id)
        if n_persisted is None:
            if path.exists():
                fmt = _detect_session_format(path)
                if fmt == "legacy":
                    _migrate_legacy_session(path)
                # After migration (or if already new format), count lines.
                try:
                    with path.open(encoding="utf-8") as f:
                        n_persisted = sum(1 for _ in f)
                except OSError:
                    n_persisted = 0
            else:
                n_persisted = 0

        new_messages = cleaned[n_persisted:]
        if new_messages:
            try:
                with path.open("a", encoding="utf-8") as f:
                    for m in new_messages:
                        f.write(json.dumps(m, ensure_ascii=False, default=str) + "\n")
            except OSError as e:
                logger.warning("session append failed for %s: %s", session_id, e)
                return
            n_persisted += len(new_messages)
            _PERSISTED_COUNTS[session_id] = n_persisted

        # Always rewrite the small sidecar so title/updated_at stay current.
        meta = {
            "session_id": session_id,
            "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "title": _session_title(cleaned),
            "n_messages": n_persisted,
        }
        tmp = idx.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            tmp.replace(idx)
        except OSError as e:
            logger.warning("session idx write failed for %s: %s", session_id, e)


def _detect_session_format(path: Path) -> str:
    """Return 'legacy' (one big JSON object on a single line wrapping
    {messages: [...]}) or 'jsonl' (one message per line)."""
    try:
        with path.open(encoding="utf-8") as f:
            first = f.readline().strip()
            if not first:
                return "jsonl"
            obj = json.loads(first)
            if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
                return "legacy"
            if isinstance(obj, dict) and obj.get("role"):
                return "jsonl"
    except (OSError, ValueError):
        pass
    return "jsonl"


def _migrate_legacy_session(path: Path) -> None:
    """Rewrite a legacy single-line session as one-message-per-line + idx."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.loads(f.read())
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    msgs = data.get("messages") or []
    if not isinstance(msgs, list):
        return
    sid = data.get("session_id") or path.stem
    tmp = path.with_suffix(".jsonl.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for m in msgs:
                f.write(json.dumps(m, ensure_ascii=False, default=str) + "\n")
        tmp.replace(path)
    except OSError as e:
        logger.warning("legacy migrate failed for %s: %s", sid, e)
        return
    # Write the index sidecar too.
    meta = {
        "session_id": sid,
        "updated_at": data.get("updated_at")
                      or _dt.datetime.now().isoformat(timespec="seconds"),
        "title": data.get("title") or _session_title(msgs),
        "n_messages": len(msgs),
    }
    idx = _idx_path(sid)
    try:
        idx.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _session_title(messages: list[dict]) -> str:
    """Use the first user message as the session label, trimmed to ~60 chars."""
    for m in messages:
        if m.get("role") == "user":
            text = (m.get("content") or "").strip()
            if not text:
                continue
            text = " ".join(text.split())
            return text[:60] + ("…" if len(text) > 60 else "")
    return "(empty)"


def list_sessions() -> list[dict]:
    """Sidebar list — newest first.

    Reads tiny `.idx.json` sidecars when available (fast); falls back to
    parsing the main `.jsonl` for legacy / un-migrated sessions.
    """
    out: list[dict] = []
    seen_ids: set[str] = set()
    if not SESSIONS_DIR.is_dir():
        return out
    # Pass 1: pick up new-format sessions via the cheap sidecar.
    for idx_p in SESSIONS_DIR.glob("*.idx.json"):
        try:
            meta = json.loads(idx_p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(meta, dict):
            continue
        sid = meta.get("session_id") or idx_p.stem.removesuffix(".idx")
        seen_ids.add(sid)
        out.append({
            "id": sid,
            "title": meta.get("title") or "(untitled)",
            "updated_at": meta.get("updated_at"),
            "n_messages": int(meta.get("n_messages") or 0),
        })
    # Pass 2: legacy or just-created sessions without a sidecar yet.
    for p in SESSIONS_DIR.glob("*.jsonl"):
        sid = p.stem
        if sid in seen_ids:
            continue
        try:
            data = _read_session_file(p)
        except Exception:  # noqa: BLE001
            continue
        out.append({
            "id": data["session_id"],
            "title": data.get("title") or _session_title(data.get("messages") or []),
            "updated_at": data.get("updated_at"),
            "n_messages": len(data.get("messages") or []),
        })
    out.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    return out


def _read_session_file(p: Path) -> dict:
    """Read a session file in EITHER format and return the unified
    {session_id, title, updated_at, messages} dict."""
    sid = p.stem
    fmt = _detect_session_format(p)
    if fmt == "legacy":
        with p.open(encoding="utf-8") as f:
            data = json.loads(f.read() or "{}")
        if not isinstance(data, dict):
            data = {}
        return {
            "session_id": data.get("session_id") or sid,
            "title": data.get("title") or _session_title(data.get("messages") or []),
            "updated_at": data.get("updated_at"),
            "messages": data.get("messages") or [],
        }
    # New format: one message per line.
    msgs: list[dict] = []
    try:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("role"):
                    msgs.append(obj)
    except OSError:
        pass
    idx = _idx_path(sid)
    title = updated_at = None
    if idx.exists():
        try:
            meta = json.loads(idx.read_text(encoding="utf-8"))
            title = meta.get("title")
            updated_at = meta.get("updated_at")
        except (OSError, ValueError):
            pass
    return {
        "session_id": sid,
        "title": title or _session_title(msgs),
        "updated_at": updated_at,
        "messages": msgs,
    }


def load_session(session_id: str) -> dict | None:
    p = SESSIONS_DIR / f"{session_id}.jsonl"
    if not p.exists():
        return None
    try:
        return _read_session_file(p)
    except Exception:  # noqa: BLE001
        return None


def delete_session(session_id: str) -> bool:
    p = SESSIONS_DIR / f"{session_id}.jsonl"
    idx = _idx_path(session_id)
    existed = p.exists() or idx.exists()
    if not existed:
        return False
    ok = True
    for q in (p, idx):
        try:
            if q.exists():
                q.unlink()
        except OSError:
            ok = False
    # Drop in-memory persist counter so a fresh persist starts clean.
    with _PERSIST_LOCK:
        _PERSISTED_COUNTS.pop(session_id, None)
    return ok


# --------------------------------------------------------------------------
# auto-memory: distill a chat session into durable facts and persist them
# in the same vector store that memory_search reads from.
# --------------------------------------------------------------------------

EXTRACT_MAX_TRANSCRIPT_CHARS = int(os.environ.get("QWEN_UI_EXTRACT_MAX_CHARS", "32000"))
EXTRACT_MAX_TOKENS = int(os.environ.get("QWEN_UI_EXTRACT_MAX_TOKENS", "1500"))
EXTRACT_MIN_USER_TURNS = int(os.environ.get("QWEN_UI_EXTRACT_MIN_TURNS", "2"))

_EXTRACT_SYSTEM = (
    "You are a precise fact-extraction service. Output ONLY valid JSON, "
    "no preamble or commentary, and no markdown fences."
)

_EXTRACT_INSTRUCTIONS = """\
Review the conversation below. Pull out DURABLE, future-useful facts the model would
want to recall in a fresh session. Be selective — fewer high-quality entries beats
many low-quality ones. Skip transient debugging chatter, clarifying questions, the
assistant's reasoning that wasn't validated, and anything that's likely to be stale
in days.

INCLUDE
  - User identity / preferences (name, role, paper authorship, tools they use).
  - Project facts that won't change soon (paths, ports, model ids, conventions).
  - Decisions made and why (chose X over Y because…).
  - Bug fixes and root causes that established a non-obvious truth.
  - Constraints learned (e.g. "dflash binds 8000, can't change").

EXCLUDE
  - Generic info / what the model said unprompted that the user didn't engage with.
  - Single-turn questions the user resolved themselves.
  - Anything obvious from the project name or tooling.

For each fact emit one object:
  {"key": "<2-5 word kebab-slug>",
   "content": "<1-3 sentences with concrete details: paths, names, numbers>",
   "tags": "<2-4 space-separated tag words>"}

If nothing is worth remembering, return {"facts": []}.

Strict output format (JSON only, no fences):
{"facts": [ {"key": "...", "content": "...", "tags": "..."}, ... ]}

CONVERSATION
============
"""


def _format_transcript_for_extraction(messages: list[dict]) -> str:
    """Render the assistant/user conversation as plain text. Tool messages
    are summarized into a one-liner so they don't dominate the budget."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        if role == "system":
            continue
        content = (m.get("content") or "").strip()
        if role == "tool":
            # Skip tool results — too verbose, rarely contains durable facts.
            # The corresponding tool_call summary in the assistant message is enough.
            continue
        prefix = {"user": "USER", "assistant": "ASSISTANT"}.get(role, role.upper())
        if role == "assistant" and m.get("tool_calls"):
            tool_names = []
            for tc in (m.get("tool_calls") or [])[:3]:
                fn = (tc.get("function") or {}).get("name") or "?"
                tool_names.append(fn)
            tool_summary = "[ran: " + ", ".join(tool_names) + "]"
            content = f"{content}\n{tool_summary}" if content else tool_summary
        if not content:
            continue
        # Per-turn cap so a single huge turn doesn't eat the budget.
        if len(content) > 3000:
            content = content[:3000] + "…"
        lines.append(f"{prefix}: {content}")
    convo = "\n\n".join(lines)
    if len(convo) > EXTRACT_MAX_TRANSCRIPT_CHARS:
        # Keep the head and the tail — facts often appear in both intros and conclusions.
        half = EXTRACT_MAX_TRANSCRIPT_CHARS // 2 - 50
        convo = convo[:half] + "\n\n…[middle of transcript truncated]…\n\n" + convo[-half:]
    return convo


def _extract_json_facts(raw: str) -> list[dict]:
    """Pull the {"facts": [...]} object out of the model's response.
    Returns the list (or [] on parse failure)."""
    if not raw:
        return []
    # Try a direct parse first.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("facts"), list):
            return obj["facts"]
    except json.JSONDecodeError:
        pass
    # Fall back to greedy bracket scan — find the first `{...}` containing "facts".
    m = re.search(r'\{[^{}]*"facts"\s*:\s*\[[^\]]*\][^{}]*\}', raw, re.DOTALL)
    if not m:
        # Looser pattern: nested brackets allowed
        m = re.search(r'\{.*?"facts"\s*:\s*\[.*?\]\s*\}', raw, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if isinstance(obj, dict) and isinstance(obj.get("facts"), list):
        return obj["facts"]
    return []


def extract_session_to_memory(session_id: str) -> tuple[int, dict]:
    """Distill a saved chat session into durable memory entries.

    Returns (status_code, payload). On success payload is
    `{"saved": N, "facts": [...], "skipped": "...", "raw": "..."}`. The
    upstream call is non-streaming and goes through the same priority
    dispatcher as chat so it doesn't fight ongoing tool calls."""
    data = load_session(session_id)
    if data is None:
        return 404, {"error": "no such session"}
    messages = data.get("messages") or []
    user_turns = sum(1 for m in messages if m.get("role") == "user")
    if user_turns < EXTRACT_MIN_USER_TURNS:
        return 200, {
            "saved": 0,
            "skipped": f"session has only {user_turns} user turn(s); need ≥ {EXTRACT_MIN_USER_TURNS}",
        }

    transcript = _format_transcript_for_extraction(messages)
    if not transcript:
        return 200, {"saved": 0, "skipped": "transcript empty after filtering"}

    user_prompt = _EXTRACT_INSTRUCTIONS + transcript

    body: dict = {
        "model": resolve_model_id(),
        "messages": [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": EXTRACT_MAX_TOKENS,
        "stream": False,
        # No tools, no thinking — keep it deterministic and short.
        "tools": [],
        "tool_choice": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }

    DISPATCHER.acquire(PRIO_NORMAL, label=f"extract:{session_id[:14]}")
    response = None
    err: Exception | None = None
    try:
        req = urllib.request.Request(
            f"{UPSTREAM_BASE}/v1/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            response = json.loads(r.read())
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        err = e
    finally:
        # Single release in finally — the previous version released in the
        # except branch AND in finally, then guarded the second call by
        # peeking at private dispatcher fields. That guard was racy:
        # between the first release and the peek, another thread could
        # acquire and become the holder, and we'd release THEIR slot.
        DISPATCHER.release()
    if err is not None:
        return 502, {"error": f"upstream call failed: {err}"}

    try:
        raw = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        return 502, {"error": f"unexpected response shape: {e}", "raw": str(response)[:600]}

    facts = _extract_json_facts(raw or "")
    if not facts:
        return 200, {"saved": 0, "facts": [], "raw": (raw or "")[:600],
                     "skipped": "model returned no parseable facts"}

    # Persist via the existing memory_save path so embeddings are produced
    # the same way and memory_search can find them.
    try:
        from agent_tools import memory_save  # lazy import — heavy
    except Exception as e:  # noqa: BLE001
        return 500, {"error": f"agent_tools.memory_save unavailable: {e}"}

    when = _dt.date.today().strftime("%Y%m%d")
    sid_short = session_id.split("-")[-1] if "-" in session_id else session_id[:6]
    saved_keys: list[str] = []
    skipped: list[str] = []

    for f in facts:
        if not isinstance(f, dict):
            continue
        key = str(f.get("key") or "").strip()
        content = str(f.get("content") or "").strip()
        tags = str(f.get("tags") or "").strip()
        if not key or not content:
            skipped.append(f"missing key/content: {f!r}"[:120])
            continue
        # Normalize key: lowercase, kebab-case, prefix with auto:
        slug = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-") or "fact"
        full_key = f"auto:{when}:{sid_short}:{slug}"[:120]
        full_tags = (tags + " auto chat-extract").strip()
        try:
            memory_save(full_key, content, full_tags)
            saved_keys.append(full_key)
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{full_key}: {type(e).__name__}: {e}"[:160])

    return 200, {
        "saved": len(saved_keys),
        "facts": saved_keys,
        "skipped": skipped,
    }


# --------------------------------------------------------------------------
# agents — config CRUD on `~/.qwen/agents/<id>/agent.json` plus a thin
# proxy to the supervisor daemon for runtime ops (run-now, cancel, status).
# Agent CRUD is filesystem-only here so the UI doesn't depend on the
# supervisor being up; the supervisor watches the directory for changes.
# --------------------------------------------------------------------------

SUPERVISOR_BASE = os.environ.get(
    "QWEN_SUPERVISOR_BASE", "http://127.0.0.1:8003"
)


def _agent_dir(aid: str) -> Path:
    return AGENTS_DIR / aid


def _agent_cfg_path(aid: str) -> Path:
    return _agent_dir(aid) / "agent.json"


def _new_agent_id() -> str:
    return uuid.uuid4().hex[:8]


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(path)


def _read_json_file(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def list_agents() -> list[dict]:
    """Read every config under ~/.qwen/agents/. Augment with live status
    from the supervisor when reachable; fall back to disk-only fields when
    not (so the UI still renders something useful)."""
    on_disk: list[dict] = []
    if AGENTS_DIR.is_dir():
        for child in sorted(AGENTS_DIR.iterdir()):
            if not child.is_dir():
                continue
            cfg = _read_json_file(child / "agent.json")
            if cfg is None:
                continue
            cfg.setdefault("id", child.name)
            on_disk.append(cfg)

    sup = _supervisor_get("/agents") or []
    by_id = {a.get("id"): a for a in sup if isinstance(a, dict)}
    out = []
    for cfg in on_disk:
        aid = cfg["id"]
        live = by_id.get(aid, {})
        out.append({
            "id": aid,
            "name": cfg.get("name") or aid,
            "enabled": cfg.get("enabled", True),
            "interval_seconds": cfg.get("interval_seconds"),
            "cwd": cfg.get("cwd"),
            "created_at": cfg.get("created_at"),
            "last_run_at": cfg.get("last_run_at"),
            "last_run_status": cfg.get("last_run_status"),
            "last_run_id": cfg.get("last_run_id"),
            "next_run_at": live.get("next_run_at"),
            "status": live.get("status") or (
                "paused" if not cfg.get("enabled", True) else "idle"
            ),
        })
    return out


def get_agent(aid: str) -> dict | None:
    cfg = _read_json_file(_agent_cfg_path(aid))
    if cfg is None:
        return None
    cfg.setdefault("id", aid)
    # supervisor may add live status / runs
    live = _supervisor_get(f"/agents/{aid}") or {}
    cfg["status"] = live.get("status") or (
        "paused" if not cfg.get("enabled", True) else "idle"
    )
    cfg["next_run_at"] = live.get("next_run_at")
    cfg["recent_runs"] = live.get("recent_runs") or _list_runs_local(aid)
    return cfg


def create_agent(payload: dict) -> tuple[int, dict]:
    name = (payload.get("name") or "").strip()
    prompt = (payload.get("prompt") or "").strip()
    if not name:
        return 400, {"error": "name required"}
    if not prompt:
        return 400, {"error": "prompt required"}
    try:
        interval = int(payload.get("interval_seconds") or 0)
    except (TypeError, ValueError):
        return 400, {"error": "interval_seconds must be int"}
    if interval < 60:
        return 400, {"error": "interval_seconds must be >= 60"}
    cwd = (payload.get("cwd") or str(Path.home())).strip()
    if not Path(cwd).expanduser().is_dir():
        return 400, {"error": f"cwd not found: {cwd}"}
    enabled = bool(payload.get("enabled", True))

    aid = _new_agent_id()
    _agent_dir(aid).mkdir(parents=True, exist_ok=True)
    cfg = {
        "id": aid,
        "name": name,
        "prompt": prompt,
        "cwd": str(Path(cwd).expanduser()),
        "interval_seconds": interval,
        "enabled": enabled,
        "created_at": time.time(),
        "last_run_at": None,
        "last_run_status": None,
        "last_run_id": None,
    }
    _atomic_write_json(_agent_cfg_path(aid), cfg)
    return 201, cfg


def update_agent(aid: str, patch: dict) -> tuple[int, dict]:
    cfg = _read_json_file(_agent_cfg_path(aid))
    if cfg is None:
        return 404, {"error": "no such agent"}
    cfg.setdefault("id", aid)
    # Whitelist: only certain fields are user-editable
    for k in ("name", "prompt", "cwd", "interval_seconds", "enabled"):
        if k in patch:
            cfg[k] = patch[k]
    # Validate name + prompt are non-empty if user is editing them.
    if "name" in patch and not str(cfg.get("name", "")).strip():
        return 400, {"error": "name required"}
    if "prompt" in patch and not str(cfg.get("prompt", "")).strip():
        return 400, {"error": "prompt required"}
    if "cwd" in patch:
        cwd_val = str(cfg.get("cwd", "")).strip()
        if not cwd_val:
            return 400, {"error": "cwd required"}
        cwd_expanded = Path(cwd_val).expanduser()
        if not cwd_expanded.is_dir():
            return 400, {"error": f"cwd not found: {cwd_val}"}
        cfg["cwd"] = str(cwd_expanded)
    if cfg.get("interval_seconds") is not None:
        try:
            cfg["interval_seconds"] = int(cfg["interval_seconds"])
        except (TypeError, ValueError):
            return 400, {"error": "interval_seconds must be int"}
        if cfg["interval_seconds"] < 60:
            return 400, {"error": "interval_seconds must be >= 60"}
    _atomic_write_json(_agent_cfg_path(aid), cfg)
    return 200, cfg


def delete_agent(aid: str) -> tuple[int, dict]:
    d = _agent_dir(aid)
    if not d.exists():
        return 404, {"error": "no such agent"}
    # cancel any in-flight run first (best effort)
    _supervisor_post(f"/agents/{aid}/cancel-run", {})
    # blow away the whole directory (config + runs)
    import shutil
    try:
        shutil.rmtree(d)
    except OSError as e:
        return 500, {"error": str(e)}
    return 200, {"deleted": True}


def _list_runs_local(aid: str, limit: int = 50) -> list[dict]:
    runs_dir = _agent_dir(aid) / "runs"
    if not runs_dir.is_dir():
        return []
    metas: list[dict] = []
    for p in sorted(runs_dir.glob("*.meta.json"), reverse=True):
        m = _read_json_file(p)
        if m is None:
            continue
        m["run_id"] = p.stem.removesuffix(".meta")
        metas.append(m)
        if len(metas) >= limit:
            break
    return metas


def get_run(aid: str, rid: str) -> dict | None:
    runs_dir = _agent_dir(aid) / "runs"
    meta = _read_json_file(runs_dir / f"{rid}.meta.json")
    if meta is None:
        return None
    log_path = runs_dir / f"{rid}.jsonl"
    events: list[dict] = []
    if log_path.exists():
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        except OSError:
            pass
    return {"meta": meta, "events": events}


# --- agent graph helpers ---------------------------------------------------

def _graphs_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "examples")


def _list_graphs() -> dict:
    """Return {graphs: [{name, file, nodes, entry_inputs}]} for /api/graphs.

    Inspects each *_graph.py for nodes and surfaces the inputs that have to
    be supplied at run time (declared on the entry node and not produced by
    any upstream). The chat UI uses this to render a graph picker.
    """
    out: list[dict] = []
    d = _graphs_dir()
    if not os.path.isdir(d):
        return {"graphs": []}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith("_graph.py"):
            continue
        name = fn[: -len("_graph.py")]
        path = os.path.join(d, fn)
        try:
            g = _load_graph(path)
            nodes = []
            for nname in g.topo_order():
                node = g.nodes[nname]
                nodes.append({
                    "name": nname,
                    "role": node.role,
                    "tools": node.tools or [],
                    "inputs": list(node.inputs),
                    "outputs": [
                        f"{n}:{t}" for (n, t) in
                        (_split_output_for_ui(o) for o in node.outputs)
                    ],
                })
            entry_inputs = _entry_inputs(g)
        except Exception as e:  # noqa: BLE001
            out.append({"name": name, "file": fn, "error": f"{type(e).__name__}: {e}"})
            continue
        out.append({"name": name, "file": fn, "nodes": nodes,
                    "entry_inputs": entry_inputs})
    return {"graphs": out}


def _split_output_for_ui(o):
    """Mirror agent_graph._split_output without taking the import dependency."""
    if isinstance(o, str):
        if ":" in o:
            n, _, t = o.partition(":")
            return (n.strip(), t.strip() or "t")
        return (o, "t")
    return (str(o[0]), str(o[1]) or "t")


def _load_graph(path: str):
    """Tolerant graph loader for the UI thread (no import-cycle drama)."""
    _ensure_scripts_on_path()
    from agent_graph import _load_graph_module  # type: ignore
    return _load_graph_module(path)


def _entry_inputs(g) -> list[str]:
    """Inputs that must be supplied at graph.run(): inputs declared by any
    node but not produced by an upstream node."""
    produced: set[str] = set()
    needed: list[str] = []
    for nname in g.topo_order():
        node = g.nodes[nname]
        for inp in node.inputs:
            if inp not in produced and inp not in needed:
                needed.append(inp)
        for o in node.outputs:
            n, _t = _split_output_for_ui(o)
            produced.add(n)
    return needed


def _list_mcps() -> dict:
    """Return the MCP registry for the UI."""
    _ensure_scripts_on_path()
    from agent_tools import _mcp_load  # type: ignore
    return _mcp_load()


def _mcp_register(body: dict) -> dict:
    _ensure_scripts_on_path()
    from agent_tools import mcp_register  # type: ignore
    return mcp_register(
        name=body.get("name") or "",
        url=body.get("url") or "",
        headers=body.get("headers") or {},
        tools=body.get("tools") or [],
    )


def _mcp_unregister(name: str) -> bool:
    _ensure_scripts_on_path()
    from agent_tools import mcp_unregister  # type: ignore
    return mcp_unregister(name)


# --------------------------------------------------------------------------
# graph runs archive — persist every run for later replay/inspection
# --------------------------------------------------------------------------

# Date-bucketed: graph_runs/YYYY-MM-DD/<run_id>.json. Listing one day's
# runs is cheap (single dir scan). Bulk pruning is `rm -rf` of an old day.
_GRAPH_RUN_RE = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")


def _new_graph_run_id() -> str:
    return _dt.datetime.now().strftime("%H%M%S-") + uuid.uuid4().hex[:8]


def _graph_run_path(date_str: str, run_id: str) -> Path:
    return GRAPH_RUNS_DIR / date_str / f"{run_id}.json"


def _persist_graph_run(record: dict) -> None:
    """Write a single graph run record into today's bucket. Best-effort —
    if the disk is full or the dir can't be created, log + skip rather
    than failing the run."""
    try:
        date_str = _dt.datetime.now().strftime("%Y-%m-%d")
        d = GRAPH_RUNS_DIR / date_str
        d.mkdir(parents=True, exist_ok=True)
        run_id = record.get("run_id") or _new_graph_run_id()
        record["run_id"] = run_id
        path = d / f"{run_id}.json"
        # Atomic write so a crash mid-write never leaves a partial file
        # that breaks the listing.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, default=str),
                       encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        logger.warning("graph run persist failed: %s", e)


def _prune_old_graph_runs() -> int:
    """Drop date-buckets older than GRAPH_RUNS_TTL_DAYS. Called lazily on
    list. Returns count of bucket dirs removed."""
    if GRAPH_RUNS_TTL_DAYS <= 0 or not GRAPH_RUNS_DIR.is_dir():
        return 0
    cutoff = _dt.date.today() - _dt.timedelta(days=GRAPH_RUNS_TTL_DAYS)
    removed = 0
    for child in GRAPH_RUNS_DIR.iterdir():
        if not child.is_dir():
            continue
        try:
            d = _dt.datetime.strptime(child.name, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < cutoff:
            try:
                shutil.rmtree(child)
                removed += 1
            except OSError as e:
                logger.warning("graph run prune of %s failed: %s", child, e)
    return removed


def list_graph_runs(*, since: str | None = None, limit: int = 50,
                     graph_filter: str | None = None) -> dict:
    """List graph runs newest-first. Pagination is by `since` (cursor =
    a `YYYY-MM-DDTHH:MM:SS` started_at; results return strictly older).

    Returns {"runs": [...], "next_cursor": str|None, "total_buckets": int}.
    Each run is a compact summary; full payload comes from /api/graph-runs/<run_id>.
    """
    _prune_old_graph_runs()
    if not GRAPH_RUNS_DIR.is_dir():
        return {"runs": [], "next_cursor": None, "total_buckets": 0}
    # Newest day first.
    bucket_dirs = sorted(
        (p for p in GRAPH_RUNS_DIR.iterdir() if p.is_dir()),
        reverse=True,
    )
    out: list[dict] = []
    next_cursor: str | None = None
    for bucket in bucket_dirs:
        # Skip whole buckets that are newer than the cursor — they were
        # already returned in a previous page.
        files = sorted(bucket.glob("*.json"), reverse=True)
        for p in files:
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            started = rec.get("started_at") or ""
            if since and started >= since:
                continue
            if graph_filter and graph_filter not in (rec.get("graph") or ""):
                continue
            # Compact summary — drop big fields that the list view doesn't need.
            summary = {
                "run_id": rec.get("run_id"),
                "graph": rec.get("graph"),
                "started_at": started,
                "wall_s": rec.get("wall_s"),
                "ok": rec.get("ok"),
                "n_nodes": len(rec.get("outputs") or {}),
                "input_keys": list((rec.get("inputs") or {}).keys()),
                "error": rec.get("error"),
            }
            out.append(summary)
            if len(out) >= limit:
                next_cursor = started
                return {
                    "runs": out, "next_cursor": next_cursor,
                    "total_buckets": len(bucket_dirs),
                }
    return {"runs": out, "next_cursor": None, "total_buckets": len(bucket_dirs)}


def get_graph_run(run_id: str) -> dict | None:
    """Find a run by id across all date buckets."""
    if not _GRAPH_RUN_RE.fullmatch(run_id) or not GRAPH_RUNS_DIR.is_dir():
        return None
    # Newest first since recent runs are more likely to be queried.
    for bucket in sorted(GRAPH_RUNS_DIR.iterdir(), reverse=True):
        if not bucket.is_dir():
            continue
        path = bucket / f"{run_id}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    return None


def delete_graph_run(run_id: str) -> bool:
    if not _GRAPH_RUN_RE.fullmatch(run_id) or not GRAPH_RUNS_DIR.is_dir():
        return False
    for bucket in GRAPH_RUNS_DIR.iterdir():
        if not bucket.is_dir():
            continue
        path = bucket / f"{run_id}.json"
        if path.exists():
            try:
                path.unlink()
                # Remove the bucket if it's now empty.
                if not any(bucket.iterdir()):
                    bucket.rmdir()
                return True
            except OSError:
                return False
    return False


def _resolve_graph(name_or_path: str):
    """Resolve a name (e.g. 'market_research') or path; raise FileNotFoundError if missing."""
    d = _graphs_dir()
    if os.path.isabs(name_or_path) and os.path.exists(name_or_path):
        return _load_graph(name_or_path), name_or_path
    candidate = os.path.join(d, f"{name_or_path}_graph.py")
    if not os.path.exists(candidate) and name_or_path.endswith(".py"):
        candidate = os.path.join(d, name_or_path)
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"no graph at {candidate!r}")
    return _load_graph(candidate), candidate


def _supervisor_get(path: str, timeout: float = 1.5) -> dict | list | None:
    try:
        req = urllib.request.Request(SUPERVISOR_BASE + path)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


def _supervisor_post(path: str, body: dict, timeout: float = 1.5) -> dict | None:
    try:
        req = urllib.request.Request(
            SUPERVISOR_BASE + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# bootstrap HTML (used until the static shell exists)
# --------------------------------------------------------------------------

_BOOTSTRAP_HTML = """<!doctype html>
<meta charset="utf-8">
<title>qwen-ui (bootstrap)</title>
<style>
  body { font: 14px system-ui; margin: 2rem; color: #222; }
  code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }
  .ok  { color: #0a7d2c; }
  .bad { color: #c0392b; }
</style>
<h1>qwen-ui</h1>
<p>Server is up. The static UI hasn't been written yet — try the API:</p>
<ul>
  <li><a href="/api/health">/api/health</a></li>
</ul>
<p id="probe">checking upstream…</p>
<script>
fetch('/api/health').then(r => r.json()).then(d => {
  const el = document.getElementById('probe');
  if (d.upstream === 'up') {
    el.innerHTML = '<span class="ok">upstream up</span> — model: <code>' + (d.model || '?') + '</code>';
  } else {
    el.innerHTML = '<span class="bad">upstream down</span> — ' + (d.error || '');
  }
});
</script>
"""


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="qwen-ui %(levelname)s: %(message)s",
    )

    _ensure_dirs()

    # Pre-warm the BGE embedder on a background thread so the first
    # memory_save / web_search call doesn't pay the 0.5-2s cold-start.
    # Disabled with QWEN_UI_PREWARM=0 (e.g. CI smoke tests where the
    # embedder isn't installed and we don't care about latency).
    if os.environ.get("QWEN_UI_PREWARM", "1") not in ("", "0", "false", "False"):
        def _prewarm():
            try:
                from agent_tools import prewarm_embedder
                err = prewarm_embedder()
                if err:
                    logger.info("embedder pre-warm skipped: %s", err)
                else:
                    logger.info("embedder pre-warm done")
            except Exception as e:  # noqa: BLE001
                logger.info("embedder pre-warm failed: %s", e)
        threading.Thread(target=_prewarm, name="prewarm-embed", daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), UIHandler)
    logger.info("listening on http://%s:%s (upstream %s)", args.host, args.port, UPSTREAM_BASE)
    logger.info("home dir: %s", UI_HOME)

    # Signal-driven shutdown: the signal handler runs on the main thread,
    # which is also where serve_forever() blocks. Calling server.shutdown()
    # from there deadlocks (shutdown() waits for serve_forever to ack, but
    # serve_forever can't return until the signal handler finishes). Use
    # a separate watcher thread that calls shutdown() once the flag flips.
    should_exit = threading.Event()

    def _on_signal(_signo=None, _frame=None):
        logger.info("signal received — shutting down")
        should_exit.set()

    def _shutdown_watcher():
        should_exit.wait()
        try:
            server.shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("server.shutdown() raised")

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    threading.Thread(target=_shutdown_watcher, name="ui-shutdown",
                     daemon=True).start()

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        # Belt + suspenders: if for any reason the signal handler didn't
        # run (some tests interrupt at an awkward time), still flag exit.
        should_exit.set()
        server.shutdown()
    finally:
        server.server_close()
        logger.info("ui exited cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())

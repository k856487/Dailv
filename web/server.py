"""Local page for Dailv 1.7. Rewrite path is dailv.main (ship CLI), not a second decoder."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from difflib import SequenceMatcher
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# Align with SHIP_NOTE before dailv.py reads these at import time.
# CLI default BON is 4; the shipped product eval uses 8.
os.environ.setdefault("DAILV_BON_N", "8")
os.environ.setdefault("DAILV_PROBE_REINJECT", "1")
os.environ.setdefault("DAILV_AGGRESSIVE_RETRY", "1")
os.environ.setdefault("DAILV_DROP_RANK", "1")
os.environ.setdefault("DAILV_AGGRESSIVE_DROP_FLOOR", "0.20")
os.environ.setdefault("DAILV_FALLBACK_SEVERE", "1")

HOST = os.environ.get("DAILV_WEB_HOST", "127.0.0.1")
PORT = int(os.environ.get("DAILV_WEB_PORT", "8780"))
MAX_CHARS = 20000
HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.html"
DAILV_PATH = Path(
    os.environ.get("DAILV_PY", str(HERE.parent / "tools" / "dailv" / "dailv.py"))
)

_lock = threading.Lock()
_busy = False
# Percent sliders. Defaults are the live recipe: drop floor 20%, ship mean similarity ~70%.
# Ranges stay near those points so a drag cannot demand a near-copy or an impossible drop.
DROP_MIN, DROP_MAX, DROP_DEFAULT = 10, 40, 20
SIM_MIN, SIM_MAX, SIM_DEFAULT = 50, 85, 70


def _load_dailv():
    import importlib.util

    name = "dailv_web_mod"
    spec = importlib.util.spec_from_file_location(name, DAILV_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {DAILV_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dailv = _load_dailv()
_BASE_17 = dailv.PROFILES["1.7b"]


def clamp_pct(value, lo: int, hi: int, default: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return min(hi, max(lo, value))


def ensure_resident() -> None:
    """Start the 1.7 llama-server when it is not already listening."""
    if dailv.resolve_server_base(_BASE_17):
        return
    if not os.path.isfile(_BASE_17.model_path):
        print(f"模型不存在: {_BASE_17.model_path}", flush=True)
        return
    cmd = dailv.build_server_cmd(_BASE_17)
    log_path = HERE / "llama-server.log"
    log = open(log_path, "a", encoding="utf-8")
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    subprocess.Popen(cmd, stdout=log, stderr=log, creationflags=flags)
    for _ in range(90):
        time.sleep(1)
        if dailv.resolve_server_base(_BASE_17):
            print(f"resident http://127.0.0.1:8765", flush=True)
            return
    print("resident 仍在加载，改写会在它就绪后复用。", flush=True)


def stream_rewrite(text: str, emit, drop_pct: int, sim_pct: int) -> None:
    """Run the ship CLI. Each finished ~200-char unit is emitted before the next starts."""
    import queue as queue_mod

    events: queue_mod.Queue = queue_mod.Queue()
    argv = ["--profile", "1.7b", "--quiet-progress", text]
    original_pace = dailv._pace_out
    drop_floor = drop_pct / 100.0
    sim_floor = sim_pct / 100.0
    saved_floor = dailv._AGGRESSIVE_DROP_FLOOR
    saved_weak = dailv._WEAK_DROP
    saved_rank = dailv._cand_rank

    def rank(src: str, out: str, *, ai_before=None):
        base = saved_rank(src, out, ai_before=ai_before)
        sim = SequenceMatcher(None, src or "", out or "").ratio()
        meet = 0 if sim + 1e-9 >= sim_floor else 1
        return (base[0], meet) + tuple(base[1:])

    def pace(chunk: str, ahead, *, steady: bool) -> None:
        if chunk:
            events.put(("segment", chunk))

    def runner() -> None:
        out = io.StringIO()
        err = io.StringIO()
        dailv._pace_out = pace
        try:
            with redirect_stdout(out), redirect_stderr(err):
                rc = dailv.main(argv)
            tail = err.getvalue().strip()[-800:]
            events.put(("end", rc, tail))
        except SystemExit as e:
            events.put(("end", 1, f"改写中断（{e}）"))
        except Exception as e:
            traceback.print_exc()
            events.put(("end", 1, str(e) or "改写失败"))
        finally:
            dailv._pace_out = original_pace

    thread = threading.Thread(target=runner, daemon=True)
    dailv._AGGRESSIVE_DROP_FLOOR = drop_floor
    dailv._WEAK_DROP = drop_floor
    dailv._cand_rank = rank
    thread.start()
    parts: list[str] = []
    try:
        emit({"type": "start"})
        while True:
            kind, *rest = events.get()
            if kind == "segment":
                parts.append(rest[0])
                emit({"type": "segment", "text": rest[0]})
                continue
            rc, tail = rest
            if rc != 0 and not parts:
                raise RuntimeError(tail or f"dailv 退出码 {rc}")
            if not parts:
                raise RuntimeError("改写结果为空")
            emit({"type": "done", "text": "".join(parts).strip(), "warning": tail if rc != 0 else ""})
            return
    finally:
        thread.join()
        dailv._AGGRESSIVE_DROP_FLOOR = saved_floor
        dailv._WEAK_DROP = saved_weak
        dailv._cand_rank = saved_rank


def status() -> dict:
    profile = dailv.PROFILES["1.7b"]
    base = dailv.resolve_server_base(profile)
    return {
        "ok": True,
        "profile": "1.7",
        "model": Path(profile.model_path).name,
        "server": base or "",
        "server_on": bool(base),
        "busy": _busy,
        "bon": int(os.environ.get("DAILV_BON_N", "8")),
        "drop_default": DROP_DEFAULT,
        "sim_default": SIM_DEFAULT,
        "drop_min": DROP_MIN,
        "drop_max": DROP_MAX,
        "sim_min": SIM_MIN,
        "sim_max": SIM_MAX,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, payload: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _begin_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _sse(self, obj: dict) -> None:
        data = f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
        self.wfile.write(b"%X\r\n" % len(data))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _end_sse(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            data = INDEX.read_bytes()
            self._send(200, data, "text/html; charset=utf-8")
            return
        if path == "/api/status":
            self._json(200, status())
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        global _busy
        path = urlparse(self.path).path
        if path != "/api/rewrite":
            self._json(404, {"ok": False, "error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > 2_000_000:
            self._json(400, {"ok": False, "error": "请求体为空或过大"})
            return
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"ok": False, "error": "JSON 无法解析"})
            return
        text = str(data.get("text") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            self._json(400, {"ok": False, "error": "请先贴入待改写段落"})
            return
        if len(text) > MAX_CHARS:
            self._json(400, {"ok": False, "error": f"单次最多 {MAX_CHARS} 字"})
            return
        drop_pct = clamp_pct(data.get("drop_pct"), DROP_MIN, DROP_MAX, DROP_DEFAULT)
        sim_pct = clamp_pct(data.get("sim_pct"), SIM_MIN, SIM_MAX, SIM_DEFAULT)
        if not _lock.acquire(blocking=False):
            self._json(409, {"ok": False, "error": "正在改写上一段，请稍候"})
            return
        _busy = True
        try:
            self._begin_sse()

            def emit(obj: dict) -> None:
                self._sse(obj)

            try:
                stream_rewrite(text, emit, drop_pct, sim_pct)
            except Exception as e:
                traceback.print_exc()
                self._sse({"type": "error", "error": str(e) or "改写失败"})
            self._end_sse()
        finally:
            _busy = False
            _lock.release()


def main() -> None:
    ensure_resident()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    info = status()
    print(f"Dailv 1.7  http://{HOST}:{PORT}", flush=True)
    print(f"model={info['model']}  BoN={info['bon']}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)


if __name__ == "__main__":
    main()

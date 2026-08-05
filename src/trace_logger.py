"""Appends one JSON object per line to logging/trace.jsonl for the current
run. README: "không append, chỉ cần lượt chạy mới nhất" — each fresh run
truncates the file first (reset_trace), then every agent step is appended
as it happens during that run.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from . import config

_lock = threading.Lock()


def reset_trace() -> None:
    config.LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    config.TRACE_PATH.write_text("", encoding="utf-8")


def log_event(
    *,
    case_id: str,
    agent: str,
    event: str,
    data: dict[str, Any] | None = None,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "agent": agent,
        "event": event,
        "data": data or {},
    }
    line = json.dumps(entry, ensure_ascii=False, default=str)
    with _lock:
        with config.TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

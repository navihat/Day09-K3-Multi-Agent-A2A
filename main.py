"""Entry point: runs the multi-agent pipeline over the case files in
input/ and writes output/<case_id>.json for each one.

Usage:
    python main.py                # all 50 cases (EC_001..EC_050)
    python main.py EC_001 EC_003  # only the given cases (for quick testing)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from src import config
from src.graph import build_graph
from src.schemas import CaseInput
from src.trace_logger import log_event, reset_trace


def _write_metadata(run_stats: dict) -> None:
    import platform
    from importlib.metadata import version as pkg_version

    metadata = {
        "model": {
            "name": config.GROQ_MODEL_NAME,
            "parameter_size": config.GROQ_MODEL_PARAMS,
            "provider": "Groq",
        },
        "framework": {
            "orchestration": "LangGraph",
            "langgraph_version": pkg_version("langgraph"),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "os": platform.platform(),
        },
        "policy_version": config.POLICY_VERSION,
        "run": run_stats,
    }
    config.LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    config.METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_case(app, case_id: str) -> tuple[bool, str | None]:
    input_path = config.INPUT_DIR / f"{case_id}.json"
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    case_input = CaseInput.model_validate(raw)

    initial_state = {
        "case_id": case_input.case_id,
        "opened_at": case_input.opened_at,
        "customer_request": case_input.customer_request.model_dump(),
        "policy_version": case_input.policy_version,
    }

    log_event(case_id=case_id, agent="runner", event="case_start")
    try:
        final_state = app.invoke(initial_state)
    except Exception as exc:  # noqa: BLE001
        log_event(case_id=case_id, agent="runner", event="case_error", data={"error": str(exc)})
        return False, str(exc)

    if final_state.get("hard_gate_failed"):
        log_event(
            case_id=case_id,
            agent="runner",
            event="hard_gate_failed",
            data={"issues": final_state.get("verifier_issues")},
        )

    output_path = config.OUTPUT_DIR / f"{case_id}.json"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(final_state["output"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_event(case_id=case_id, agent="runner", event="case_done", data={"output_path": str(output_path)})
    return not final_state.get("hard_gate_failed", False), None


def main() -> None:
    load_dotenv()
    reset_trace()

    requested = sys.argv[1:]
    if requested:
        case_ids = requested
    else:
        case_ids = [f"EC_{i:03d}" for i in range(1, 51)]

    app = build_graph()

    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()

    ok_count = 0
    failed: list[dict] = []
    for case_id in case_ids:
        print(f"[{case_id}] running...", flush=True)
        ok, error = run_case(app, case_id)
        if ok:
            ok_count += 1
            print(f"[{case_id}] done")
        else:
            failed.append({"case_id": case_id, "error": error})
            print(f"[{case_id}] FAILED: {error}")

    elapsed = round(time.monotonic() - start, 1)
    run_stats = {
        "started_at": started_at,
        "cases_requested": len(case_ids),
        "cases_ok": ok_count,
        "cases_failed": failed,
        "elapsed_seconds": elapsed,
    }
    _write_metadata(run_stats)
    print(f"\n{ok_count}/{len(case_ids)} cases OK in {elapsed}s. metadata.json updated.")


if __name__ == "__main__":
    main()

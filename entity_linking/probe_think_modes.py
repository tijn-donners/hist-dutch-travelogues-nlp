#!/usr/bin/env python3
"""Controlled probe of Ollama think-mode token output and server-side stats.

Sends ONE fixed disambiguation prompt to each model at each think mode
(false / low / medium / high) and records, per call:

  - client wall time and internal retry count
  - length of the thinking trace and the content
  - whether the thinking-only fallback fired
  - the final chunk's server-side stats from the Ollama API:
    eval_count (generated tokens incl. thinking), eval_duration,
    total_duration, load_duration, prompt_eval_count,
    prompt_eval_duration, done_reason (ns -> ms where applicable)

This is meant to settle why the legacy thinking-mode EL runs show a duration
inversion (low > medium > high) without running the full 315-entity pipeline.
It calls ``ollama_utils.stream_ollama_chat`` directly with its (non-breaking)
``stats`` hook, so the EL pipeline is untouched.

Usage (from anywhere):
    OLLAMA_API_KEY=... python entity_linking/probe_think_modes.py
    python entity_linking/probe_think_modes.py --models gemma4:31b --think low high

Output: table to stdout + appended rows to
    entity_linking/el-evaluation/probe_think_modes.csv
"""

import argparse
import csv
import os
import time
from pathlib import Path

from dotenv import load_dotenv

# Make ollama_utils importable regardless of cwd (repo root is parent of this dir).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ollama_utils import stream_ollama_chat  # noqa: E402

# Load OLLAMA_API_KEY (and any other vars) from the repo .env, mirroring el.py.
# Use an explicit path so it resolves regardless of the caller's cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DEFAULT_MODELS = ["deepseek-v4-flash", "gemma4:31b", "kimi-k2.7-code"]
THINK_MODES = ["false", "low", "medium", "high"]
# "false" must reach the API as a bool (Ollama accepts bool|'low'|'medium'|'high').
THINK_VALUES = {"false": False, "low": "low", "medium": "medium", "high": "high"}

# Match the legacy runs' corrected EPG temperature (0.1) so the only varying
# factor across calls is `think`.
TEMPERATURE = 0.1
CALL_TIMEOUT = 600.0

OUT_CSV = Path(__file__).resolve().parent / "el-evaluation" / "probe_think_modes.csv"

COLUMNS = [
    "timestamp", "run_id", "model", "think", "wall_seconds", "retries",
    "thinking_chars", "content_chars", "fallback_used", "eval_count",
    "eval_duration_ms", "total_duration_ms", "load_duration_ms",
    "prompt_eval_count", "prompt_eval_duration_ms", "done_reason", "error",
]

# One fixed, self-contained disambiguation prompt. Reused verbatim across all
# calls so `think` is the only varying factor. A genuine Flanders/northern-France
# disambiguation gives a thinking model something to reason about.
PROMPT = """Entity mention: "Cassel"
Context: "Te Cassel aangekomen, bezocht hij de markt op de grote plaets, en vertrok den volgenden dag richting Rijsel."

Candidate Wikidata entities:
1. Q652511 — Cassel, commune in the Nord department, Hauts-de-France, France (a historic hilltop town in French Flanders).
2. Q301803 — Kassel, town in Hesse, Germany.

The source is a 19th-century Dutch travelogue describing travel through Flanders and northern France. Which Wikidata entity does "Cassel" refer to?

Reason briefly, then end your answer with exactly one line of the form:
ANSWER: Q<id>
"""


def resolve_host_and_headers():
    """Mirror el.py's cloud/local resolution (el.py:49-54)."""
    if os.environ.get("OLLAMA_API_KEY"):
        url = "https://ollama.com"
        headers = {"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"}
    else:
        url = "http://localhost:11434"
        headers = None
    api_key = None
    if headers:
        auth = headers.get("Authorization", "")
        api_key = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else None
    return url, api_key


def run_one(model, think_mode, host, api_key, run_id):
    """Call stream_ollama_chat once with a stats dict; return a result row."""
    from datetime import datetime
    stats = {}
    row = {col: "" for col in COLUMNS}
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    row["run_id"] = run_id
    row["model"] = model
    row["think"] = think_mode
    try:
        stream_ollama_chat(
            model=model,
            prompt=PROMPT,
            host=host,
            api_key=api_key,
            timeout=CALL_TIMEOUT,
            temperature=TEMPERATURE,
            think=THINK_VALUES[think_mode],
            stats=stats,
        )
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        # Still record whatever stats were captured before the failure.
    for k in COLUMNS:
        if k in stats and row[k] == "":
            row[k] = stats[k]
    return row


def write_rows(rows, out_csv):
    """Append rows to the CSV, migrating an older schema in place if needed.

    If the file exists with a header that doesn't match COLUMNS (e.g. from
    before timestamp/run_id were added), rewrite it with the new header and
    blank-pad the missing columns on existing rows so nothing is lost.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
            old_fields = reader.fieldnames
        if old_fields is not None and list(old_fields) != COLUMNS:
            # Schema drift: rewrite the whole file with the current schema,
            # preserving old rows with missing columns blanked.
            for r in existing:
                for c in COLUMNS:
                    r.setdefault(c, "")
            with open(out_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                for r in existing:
                    writer.writerow(r)
                for r in rows:
                    writer.writerow(r)
            return

    write_header = not out_csv.exists()
    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Ollama model names (default: {DEFAULT_MODELS})")
    parser.add_argument("--think", nargs="+", default=THINK_MODES,
                        choices=THINK_MODES,
                        help=f"think modes to probe (default: {THINK_MODES})")
    args = parser.parse_args()

    host, api_key = resolve_host_and_headers()
    where = "ollama.com (cloud)" if api_key else "localhost"
    # One run_id per invocation so repeated runs stay distinguishable in the CSV.
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Probing {len(args.models)} model(s) x {len(args.think)} think mode(s) "
          f"via {where}  (run_id={run_id})\n")

    rows = []
    for model in args.models:
        for think_mode in args.think:
            print(f"=== {model} | think={think_mode} ===")
            t0 = time.time()
            row = run_one(model, think_mode, host, api_key, run_id)
            rows.append(row)
            print(f"  done in {time.time() - t0:.1f}s "
                  f"(call wall={row['wall_seconds']}s, retries={row['retries']}, "
                  f"eval_count={row['eval_count']}, "
                  f"thinking_chars={row['thinking_chars']}, "
                  f"done_reason={row['done_reason']}"
                  + (f", ERROR={row['error']}" if row['error'] else "") + ")\n")

    write_rows(rows, OUT_CSV)
    print(f"Appended {len(rows)} row(s) to {OUT_CSV}")

    # Print a compact summary table.
    print("\nSummary:")
    print(f"{'model':<20}{'think':<8}{'wall_s':>9}{'retries':>8}"
          f"{'eval_count':>11}{'think_ch':>10}{'load_ms':>9}{'done_reason':>13}")
    for r in rows:
        print(f"{r['model']:<20}{r['think']:<8}{str(r['wall_seconds']):>9}"
              f"{str(r['retries']):>8}{str(r['eval_count']):>11}"
              f"{str(r['thinking_chars']):>10}{str(r['load_duration_ms']):>9}"
              f"{str(r['done_reason']):>13}")


if __name__ == "__main__":
    main()
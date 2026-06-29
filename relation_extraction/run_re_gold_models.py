"""Run every NER-stage model through RE on the gold-standard EL output.

This is the "RE-only" evaluation harness: NER and EL are fixed to the gold
standard (``1816_el_gs_el.spacy``), so the only variable is the RE model.
Every model runs at temperature 0.0 with its **default** thinking mode (no
``--think`` flag is passed → ``_thinkDefault``), then ``re_evaluate.py`` scores
it against ``1816_re_gs.csv`` and appends a row to ``re-evaluation/scores.csv``.

Model set: all models used in the NER stage (see ner/ner_and_eval_all_fewshot.sh),
minus ``cogito-2.1:671b`` (no longer available on the Ollama API), plus
``glm-5.2`` alongside ``glm-5.1``.

The script is idempotent and resumable: a model whose RE output (events.json +
.ttl) already exists is skipped, and a model already scored in scores.csv is not
re-evaluated. Pass ``--force`` to re-run everything.

Usage:
    python relation_extraction/run_re_gold_models.py
    python relation_extraction/run_re_gold_models.py --force
    python relation_extraction/run_re_gold_models.py --only glm-5.1 glm-5.2
    python relation_extraction/run_re_gold_models.py --host http://localhost:1344

The Ollama host follows rel_extraction.py's default (cloud when
OLLAMA_API_KEY is set, else localhost:11434); override with ``--host`` or the
OLLAMA_HOST env var.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

# ── CONFIGURE THIS ────────────────────────────────────────────────────
# All NER-stage models, minus cogito (removed from the Ollama API), plus
# glm-5.2 alongside glm-5.1.
MODELS = [
    "gemma4:31b",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "mistral-large-3:675b",
    "qwen3.5:397b",
    "kimi-k2.7-code",
    "glm-5.1",
    "glm-5.2",
]

TEMPERATURE = 0.0          # all models run at t=0.0
# No --think flag → model default thinking mode → "_thinkDefault" slug.
# ───────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RE_SCRIPT = Path(__file__).resolve().parent / "rel_extraction.py"
EVAL_SCRIPT = Path(__file__).resolve().parent / "re_evaluate.py"
GOLD_EL_SPACY = ROOT / "entity_linking" / "el-results" / "1816_el_gs_el.spacy"
GOLD_RE_CSV = Path(__file__).resolve().parent / "1816_re_gs.csv"
OUTPUT_DIR_RE = ROOT / "output" / "re"
OUTPUT_DIR_RDF = ROOT / "output" / "rdf"
SCORES_CSV = Path(__file__).resolve().parent / "re-evaluation" / "scores.csv"

# The gold EL input stem (after stripping the trailing "_el").
INPUT_STEM = "1816_el_gs"


def model_slug(model: str) -> str:
    """Slug a model name exactly the way rel_extraction.py does, so the
    output filenames we check match the ones rel_extraction produces."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', model)


def output_stem(model: str) -> str:
    """Expected output stem, e.g. 1816_el_gs__gemma4_31b_t0.0_thinkDefault."""
    return f"{INPUT_STEM}__{model_slug(model)}_t{TEMPERATURE}_thinkDefault"


def extraction_done(model: str) -> bool:
    """True if both RE outputs (events.json + .ttl) already exist."""
    stem = output_stem(model)
    return (OUTPUT_DIR_RE / f"{stem}_events.json").exists() and \
           (OUTPUT_DIR_RDF / f"{stem}_events.ttl").exists()


def already_scored(model: str) -> bool:
    """True if scores.csv already has a row for this model at t=0.0/default."""
    if not SCORES_CSV.exists():
        return False
    with open(SCORES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("model") == model
                    and row.get("temperature") == "0.0"
                    and row.get("think_mode") == "default"
                    and row.get("source_text") == INPUT_STEM):
                return True
    return False


def read_duration(model: str) -> float:
    """Read duration_seconds from the run's meta JSON, else 0.0."""
    meta = OUTPUT_DIR_RE / f"{output_stem(model)}_events.meta.json"
    if meta.exists():
        try:
            with open(meta) as f:
                return float(json.load(f).get("duration_seconds", 0.0) or 0.0)
        except (json.JSONDecodeError, OSError, ValueError):
            return 0.0
    return 0.0


def run_extraction(model: str, host: str | None) -> int:
    stem = output_stem(model)
    print(f"\n{'=' * 60}\n  RE extraction: {model}\n  -> {stem}_events.json/.ttl\n{'=' * 60}")
    cmd = [
        sys.executable, str(RE_SCRIPT),
        "--model", model,
        "--temperature", str(TEMPERATURE),
        "--input", str(GOLD_EL_SPACY),
    ]
    if host is not None:
        cmd += ["--host", host]
    return subprocess.run(cmd, cwd=ROOT).returncode


def run_evaluation(model: str) -> int:
    stem = output_stem(model)
    print(f"\n{'=' * 60}\n  RE evaluation: {model}\n  -> scores.csv (+ metadata)\n{'=' * 60}")
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--events", str(OUTPUT_DIR_RE / f"{stem}_events.json"),
        "--mention-map", str(OUTPUT_DIR_RE / f"{stem}_mention_map.json"),
        "--rdf", str(OUTPUT_DIR_RDF / f"{stem}_events.ttl"),
        "--gold", str(GOLD_RE_CSV),
        "--duration", str(read_duration(model)),
        "--inference-type", "cloud",
        # model / temperature / think / source-text auto-detected from the TTL
        # filename by re_evaluate.py (single source of truth).
    ]
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=None,
                        help="Ollama host forwarded to rel_extraction.py "
                             "(e.g. http://localhost:1344). Default: env/auto.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Restrict to a subset of models (space-separated).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run extraction AND evaluation even if outputs "
                             "already exist (overwrites RE outputs; appends a new "
                             "scores.csv row).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan (which models need extraction / eval) "
                             "and exit without invoking the LLM or the evaluator.")
    args = parser.parse_args()

    models = args.only if args.only else MODELS
    unknown = [m for m in models if m not in MODELS] if args.only else []
    if unknown:
        print(f"Unknown model(s) in --only (not in MODELS): {unknown}")
        sys.exit(1)

    if not GOLD_EL_SPACY.exists():
        print(f"Gold EL input not found: {GOLD_EL_SPACY}")
        sys.exit(1)
    if not GOLD_RE_CSV.exists():
        print(f"Gold RE CSV not found: {GOLD_RE_CSV}")
        sys.exit(1)

    # Plan
    print(f"RE-only gold-standard run — {len(models)} model(s), t={TEMPERATURE}, "
          f"default thinking mode.\n")
    plan = []
    for m in models:
        need_extract = args.force or not extraction_done(m)
        need_eval = args.force or not already_scored(m)
        # If extraction is missing, eval can't run until it's produced.
        if not need_extract and need_eval:
            pass  # eval an existing run
        if need_extract:
            need_eval = True  # fresh extraction → must (re)score
        plan.append((m, need_extract, need_eval))
        ex = "EXTRACT" if need_extract else ("exists " if not args.force else "EXTRACT*")
        ev = "EVAL" if need_eval else ("scored " if not args.force else "EVAL*")
        print(f"  {m:<26} -> {ex:<8} {ev}")

    todo = [(m, ne, nv) for (m, ne, nv) in plan if ne or nv]
    if not todo:
        print("\nAll models already extracted and scored. Nothing to do "
              "(use --force to re-run).")
        return
    print(f"\n{len(todo)} model(s) have work pending.\n")

    if args.dry_run:
        print("Dry run — no extraction or evaluation invoked.")
        return

    failures = []
    for m, need_extract, need_eval in plan:
        if not (need_extract or need_eval):
            continue
        if need_extract:
            rc = run_extraction(m, args.host)
            if rc != 0:
                print(f"\n  ❌ EXTRACTION FAILED (exit {rc}): {m}\n")
                failures.append((m, "extraction", rc))
                continue
            print(f"\n  ✅ extraction done: {m}\n")
        if need_eval:
            if not extraction_done(m):
                print(f"\n  ⏭️  skip eval (no extraction output): {m}\n")
                failures.append((m, "no-output", None))
                continue
            rc = run_evaluation(m)
            if rc != 0:
                print(f"\n  ❌ EVAL FAILED (exit {rc}): {m}\n")
                failures.append((m, "evaluation", rc))
            else:
                print(f"\n  ✅ eval done: {m}\n")

    print("\n" + "=" * 60)
    if failures:
        print(f"Completed with {len(failures)} failure(s):")
        for m, stage, rc in failures:
            print(f"  ❌ {m:<26} {stage}" + (f" (exit {rc})" if rc is not None else ""))
        sys.exit(1)
    print(f"All done — {len(models)} model(s). Scores appended to: {SCORES_CSV}")


if __name__ == "__main__":
    main()
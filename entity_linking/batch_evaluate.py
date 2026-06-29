#!/usr/bin/env python3
"""Batch-evaluate all EL results against the gold standard.

Iterates over every ``*_el.spacy`` file in ``el-results/``, reads the
matching ``*_run_info.json`` for metadata (model, temperature, think mode,
top-k, duration, pipeline stats), and calls ``el_evaluate.py`` for each.

The gold-standard reference file (``{source}_el.spacy``, with no ``__model``
segment) is skipped — it's the source the gold CSV was built from, not a
model run, so evaluating it would score the gold against itself.

Files whose stem starts with ``1816_el_gs__`` are evaluated against the
default gold CSV (``1816_el_gs.csv``).  Other files use ``--gold`` to point
at a gold CSV derived from the source-text prefix of the filename.

Usage::

    python entity_linking/batch_evaluate.py
    python entity_linking/batch_evaluate.py --dry-run   # just print what would run
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EL_RESULTS_DIR = SCRIPT_DIR / "el-results"
EL_EVALUATE = SCRIPT_DIR / "el_evaluate.py"
DEFAULT_GOLD = SCRIPT_DIR / "1816_el_gs.csv"
# Use venv Python if available (has spacy installed)
_VENV_PYTHON = SCRIPT_DIR.parent / ".venv" / "bin" / "python3"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable


def find_run_info(el_spacy_path: Path) -> Path | None:
    """Find the matching ``*_run_info.json`` for an ``*_el.spacy`` file."""
    stem = el_spacy_path.stem
    # Remove trailing _el
    if stem.endswith("_el"):
        base = stem[:-3]
    else:
        base = stem
    # Try exact match
    candidate = el_spacy_path.parent / f"{base}_run_info.json"
    if candidate.exists():
        return candidate
    # Try glob
    candidates = sorted(el_spacy_path.parent.glob(f"{base}*run_info*"))
    if candidates:
        return candidates[0]
    return None


def _parse_filename_metadata(stem: str) -> dict:
    """Parse model, temperature, and think mode from a filename stem.

    Handles naming patterns like::

        {source}__{model}_t{temp}[_think{mode}]
        {source}__{ner_model}_t{temp}_{ner_config}__{el_model}_t{temp2}[_think{mode}]

    In filenames, ``:`` in model names is replaced with ``-`` (OS-safe),
    so ``gemma4:31b`` becomes ``gemma4-31b``.  We convert the most likely
    ``{family}-{version}`` pattern back to ``{family}:{version}``.
    """
    # Strip trailing _el
    if stem.endswith("_el"):
        stem = stem[:-3]

    # Take the last __-delimited segment — that holds the EL model info
    if "__" in stem:
        last = stem.split("__")[-1]
    else:
        last = stem

    # Find temperature anchor like _t0.0, _t1.0, _t0.1
    m = re.search(r"_t(\d+\.?\d*)", last)
    if m:
        temperature = float(m.group(1))
        model_part = last[:m.start()]  # everything before _t...
        # Remaining after _t{temp}
        after_temp = last[m.end():]
        think = ""
        think_m = re.search(r"_think(\w+)", after_temp)
        if think_m:
            think = think_m.group(1)
    else:
        temperature = 0.1
        model_part = last
        think = ""

    # Restore colon for model family:version patterns
    # (e.g. gemma4-31b → gemma4:31b, kimi-k2.7-code stays as-is)
    # Heuristic: if the last hyphen-separated part starts with a digit, it's a version
    if "-" in model_part:
        parts = model_part.rsplit("-", 1)
        if parts[1] and parts[1][0].isdigit():
            model_part = f"{parts[0]}:{parts[1]}"

    return {"model": model_part, "temperature": temperature, "think": think}


def find_offset_map(el_spacy_path: Path) -> Path | None:
    """Find the matching offset map for an EL .spacy file."""
    stem = el_spacy_path.stem
    if stem.endswith("_el"):
        base = stem[:-3]
    else:
        base = stem
    candidate = el_spacy_path.parent / f"{base}_offset_map.json"
    if candidate.exists():
        return candidate
    candidates = sorted(el_spacy_path.parent.glob(f"{base}*offset_map*"))
    if candidates:
        return candidates[0]
    return None


def infer_gold_csv(el_spacy_path: Path) -> Path:
    """Infer the gold CSV path from the filename stem.

    For filenames like ``1816_el_gs__gemma4-31b_t0.0_el.spacy`` the source
    text is ``1816_el_gs`` and the gold CSV is ``1816_el_gs.csv``.

    For filenames like ``1816_third_letter__..._el.spacy`` the source text
    is ``1816_third_letter`` and we look for ``1816_third_letter_el_gs.csv``
    or fall back to the default.
    """
    stem = el_spacy_path.stem
    # Remove trailing _el
    if stem.endswith("_el"):
        base = stem[:-3]
    else:
        base = stem
    # The source text is the part before the first __ (if any)
    if "__" in base:
        source_text = base.split("__")[0]
    else:
        source_text = base.split("_el")[0]

    # Try source-specific gold CSV
    gold = SCRIPT_DIR / f"{source_text}.csv"
    if gold.exists():
        return gold
    # Try with _el_gs suffix
    gold = SCRIPT_DIR / f"{source_text}_el_gs.csv"
    if gold.exists():
        return gold
    return DEFAULT_GOLD


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate all EL results against gold standard."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    args = parser.parse_args()

    spacy_files = sorted(EL_RESULTS_DIR.glob("*_el.spacy"))
    if not spacy_files:
        print(f"No *_el.spacy files found in {EL_RESULTS_DIR}")
        sys.exit(1)

    print(f"Found {len(spacy_files)} EL result files\n")

    for spacy_path in spacy_files:
        print(f"{'=' * 70}")
        print(f"  {spacy_path.name}")
        print(f"{'=' * 70}")

        # Skip the gold-standard reference .spacy. Model outputs are named
        # ``{source}__{model}_t{temp}..._el.spacy`` (note the ``__`` separating
        # source from model); the bare reference file is just ``{source}_el.spacy``
        # — the source the gold CSV was built from, not a model run. Evaluating
        # it would score the gold against itself.
        base_stem = spacy_path.stem[:-3] if spacy_path.stem.endswith("_el") else spacy_path.stem
        if "__" not in base_stem:
            print("  ⊘ gold-reference file (not a model run) — skipping\n")
            continue

        # Check offset map
        offset_map = find_offset_map(spacy_path)
        if not offset_map:
            print(f"  ⚠ No offset map found — skipping\n")
            continue

        # Extract source text from filename (everything before the first __)
        stem = spacy_path.stem
        if stem.endswith("_el"):
            stem = stem[:-3]
        source_text = stem.split("__")[0] if "__" in stem else stem

        # Check run info
        run_info_path = find_run_info(spacy_path)
        if run_info_path:
            with open(run_info_path) as f:
                info = json.load(f)
            model = info.get("model", "")
            temperature = info.get("temperature", 0.1)
            top_k = info.get("top_k", 3)
            think = info.get("think", "")
            duration = info.get("duration", 0.0)
            pipeline_stats = info.get("stats", {})
            inference_type = info.get("inference_type",
                                      "cloud" if "cloud" in model else "local")
            print(f"  Model:        {model}")
            print(f"  Temperature:  {temperature}")
            print(f"  Think:        {think or '—'}")
            print(f"  Top-k:        {top_k}")
            print(f"  Duration:     {duration:.0f}s")
        else:
            print(f"  ⚠ No run_info.json found — parsing filename")
            meta = _parse_filename_metadata(spacy_path.stem)
            model = meta["model"]
            temperature = meta["temperature"]
            think = meta["think"]
            top_k = 3
            duration = 0.0
            pipeline_stats = {}
            inference_type = "cloud" if "cloud" in model else "local"
            print(f"  Model:        {model}  (from filename)")
            print(f"  Temperature:  {temperature}  (from filename)")
            print(f"  Think:        {think or '—'}  (from filename)")
            print(f"  Top-k:        {top_k}  (default)")
            print(f"  Duration:     —  (run info missing)")

        # Determine gold CSV
        gold_csv = infer_gold_csv(spacy_path)
        if not gold_csv.exists():
            print(f"  ⚠ Gold CSV not found: {gold_csv} — skipping\n")
            continue
        print(f"  Gold CSV:     {gold_csv.name}")
        print(f"  Source text:  {source_text}")

        # Build command
        cmd = [
            PYTHON, str(EL_EVALUATE),
            "--gold", str(gold_csv),
            "--el", str(spacy_path),
            "--model", model,
            "--temperature", str(temperature),
            "--top-k", str(top_k),
            "--duration", str(duration),
            "--pipeline-stats", json.dumps(pipeline_stats),
            "--inference-type", inference_type,
            "--source-text", source_text,
        ]
        if think:
            cmd += ["--think", str(think)]

        if args.dry_run:
            print(f"  Would run: {' '.join(cmd)}")
        else:
            print(f"  Running evaluation...")
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                print(f"  ❌ Evaluation failed (exit code {result.returncode})")
            else:
                print(f"  ✅ Done")
        print()

    print(f"{'=' * 70}")
    print(f"  Batch evaluation complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

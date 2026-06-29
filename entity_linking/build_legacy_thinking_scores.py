#!/usr/bin/env python3
"""Salvage the legacy thinking-mode EL runs from ``el-results-bugged-run/``.

The runs in ``el-results-bugged-run/`` were produced before the temperature-chain
bug was fixed. Per that folder's README, the slug filenames **and** the
``temperature`` metadata are wrong: the EPG (candidate-generation) LLM actually
ran at 0.1 and the reranker / selector LLMs at 0.0, but the recorded temperature
is 1.0 and the filenames encode ``_t1.0``. Everything else (duration, stats) is
valid. The thinking-mode runs (low/medium/high) are expensive to reproduce, so
we keep them and only correct the bookkeeping; the think=false runs are ignored.

This script is **idempotent** and does three things, in order:

1. Rename the 9 thinking-mode runs (3 models x {low,medium,high}, 4 sibling
   files each = 36 files) from ``_t1.0_`` to ``_t0.1_``.
2. Correct the 9 ``*_run_info.json``: set ``temperature = 0.1`` and add the
   per-stage fields ``epg_temp = 0.1``, ``reranker_temp = 0.0``,
   ``selection_temp = 0.0``. (``temperature`` is kept and corrected so existing
   readers / ``el_evaluate`` auto-discovery stay accurate.)
3. Build ``el-evaluation/scores_legacy_thinking.csv`` (same schema as the newest
   ``scores.csv``) by recomputing metrics from the existing ``*_el.spacy``
   predictions via the reused ``el_evaluate`` helpers -- **no re-inference**.

Run from the ``entity_linking/`` directory (or anywhere; paths are absolute via
``__file__`` and ``el_evaluate``'s constants).
"""

import json
import os
import sys
from pathlib import Path

# Ensure the script's own directory is importable (so `import el_evaluate` works
# regardless of the current working directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import el_evaluate as ev  # noqa: E402
import spacy  # noqa: E402
from spacy.tokens import DocBin  # noqa: E402

BUGGED_DIR = Path(__file__).resolve().parent / "el-results-bugged-run"
OUT_CSV = ev.EL_EVAL_DIR / "scores_legacy_thinking.csv"
THINK_MODES = ("low", "medium", "high")
SIBLING_SUFFIXES = ("_el.spacy", "_el_run_info.json", "_el_el_eval.csv", "_offset_map.json")


def _is_thinking_mode(name: str) -> bool:
    return any(f"_think{mode}_" in name or name.endswith(f"_think{mode}") for mode in THINK_MODES)


def rename_thinking_runs() -> int:
    """Rename the 36 thinking-mode sibling files ``_t1.0_`` -> ``_t0.1_``."""
    n = 0
    for spacy_path in sorted(BUGGED_DIR.glob("*_el.spacy")):
        if not _is_thinking_mode(spacy_path.name):
            continue
        stem = spacy_path.stem
        base = stem[:-3] if stem.endswith("_el") else stem
        for suf in SIBLING_SUFFIXES:
            old = spacy_path.parent / f"{base}{suf}"
            if not old.exists() or "_t1.0_" not in old.name:
                continue
            new = spacy_path.parent / old.name.replace("_t1.0_", "_t0.1_")
            os.replace(old, new)
            n += 1
            print(f"  rename: {old.name} -> {new.name}")
    return n


def fix_metadata() -> int:
    """Correct temperature + add per-stage temp fields in the 9 run_info.json."""
    n = 0
    for ri in sorted(BUGGED_DIR.glob("*_el_run_info.json")):
        if not _is_thinking_mode(ri.name):
            continue
        with open(ri, encoding="utf-8") as f:
            data = json.load(f)
        data["temperature"] = 0.1
        data["epg_temp"] = 0.1
        data["reranker_temp"] = 0.0
        data["selection_temp"] = 0.0
        with open(ri, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        n += 1
        print(f"  metadata fixed: {ri.name}  "
              f"(model={data.get('model')}, think={data.get('think')}, "
              f"duration={data.get('duration')})")
    return n


def build_scores() -> int:
    """Recompute metrics from the existing .spacy files -> scores_legacy_thinking.csv."""
    if OUT_CSV.exists():
        OUT_CSV.unlink()  # fresh file -> _append_scores_csv writes a clean header

    gold_rows = ev.load_gold_csv(ev.DEFAULT_GOLD_CSV)
    nlp = spacy.blank("nl")
    n = 0

    for spacy_path in sorted(BUGGED_DIR.glob("*_el.spacy")):
        if not _is_thinking_mode(spacy_path.name):
            continue

        run_info_path = ev._find_run_info(spacy_path)
        offset_map_path = ev.find_offset_map(spacy_path)
        if run_info_path is None or not run_info_path.exists():
            print(f"  SKIP (no run_info): {spacy_path.name}")
            continue
        if offset_map_path is None or not offset_map_path.exists():
            print(f"  SKIP (no offset_map): {spacy_path.name}")
            continue

        with open(run_info_path, encoding="utf-8") as f:
            info = json.load(f)
        with open(offset_map_path, encoding="utf-8") as f:
            offset_map = json.load(f)

        boundaries = ev.build_page_boundaries(offset_map)
        pred_docs = list(DocBin().from_disk(str(spacy_path)).get_docs(nlp.vocab))
        gold_by_page = ev.assign_gold_to_pages(gold_rows, boundaries)
        page_order = [b[0] for b in boundaries]
        el_metrics, full_metrics, _ = ev.evaluate(
            gold_by_page, pred_docs, boundaries, page_order,
        )

        ev._append_scores_csv(
            path=OUT_CSV,
            source_text=spacy_path.name.split("__")[0],  # "1816_el_gs"
            model=info["model"],
            temperature=info["temperature"],              # 0.1 (corrected)
            top_k_rerank=info.get("top_k", 3),
            inference_type=info.get("inference_type", "cloud"),
            total_entities=len(gold_rows),
            duration_seconds=info.get("duration", 0.0),
            pipeline_stats=info.get("stats", {}),
            wd=full_metrics["wd"],
            gn=full_metrics["gn"],
            combined=full_metrics["combined"],
            nil=el_metrics["nil"],
            nil_wd=el_metrics["nil_wd"],
            nil_gn=el_metrics["nil_gn"],
            think=info.get("think", ""),
        )
        n += 1
        print(f"  scored: {spacy_path.name}  "
              f"(wd_f1={full_metrics['wd'].f1:.4f}, "
              f"gn_f1={full_metrics['gn'].f1:.4f}, "
              f"nil_wd_f1={el_metrics['nil_wd'].f1:.4f})")
    return n


def main() -> None:
    print("=" * 70)
    print("Salvage legacy thinking-mode EL runs")
    print("=" * 70)

    print("\n1. Renaming _t1.0_ -> _t0.1_ (thinking-mode runs only)...")
    n_renamed = rename_thinking_runs()
    print(f"   {n_renamed} file(s) renamed.")

    print("\n2. Fixing run_info.json metadata...")
    n_meta = fix_metadata()
    print(f"   {n_meta} metadata file(s) fixed.")

    print("\n3. Building scores_legacy_thinking.csv (recompute from .spacy)...")
    n_scored = build_scores()
    print(f"   {n_scored} run(s) scored -> {OUT_CSV}")

    print("\nDone.")


if __name__ == "__main__":
    main()
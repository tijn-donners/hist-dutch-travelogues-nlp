#!/usr/bin/env python3
"""Evaluate EL predictions against gold standard KB IDs.

Loads a gold-standard CSV (with wikidata_qid and geonames_id columns manually
filled by the user) and an EL-predicted .spacy file + offset map. Computes
precision/recall/F1 for Wikidata QID prediction and GeoNames ID prediction
separately, using relaxed span matching (overlap-based) between gold and
predicted entities.

Two levels of reporting:
  1. EL-only:  conditioned on predicted entities that overlap a gold span
               (isolates linking accuracy from NER detection errors).
  2. Full:     end-to-end, including entities NER missed entirely.

Usage:
    python entity_linking/el_evaluate.py
    python entity_linking/el_evaluate.py --gold ner/gs_el_gold_template.csv \\
                                         --el entity_linking/el-results/FILE_el.spacy
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import spacy
from spacy.tokens import Span, Token, DocBin

# ── Register custom extensions (must match el.py) ──────────────────────────────
for attr in ("kb_id_wikidata_", "kb_id_geonames_"):
    if not Span.has_extension(attr):
        Span.set_extension(attr, default=None)
    if not Token.has_extension(attr):
        Token.set_extension(attr, default=None)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
NER_RESULTS_DIR = ROOT_DIR / "ner" / "ner-output"
EL_RESULTS_DIR = SCRIPT_DIR / "el-results"
EL_EVAL_DIR = SCRIPT_DIR / "el-evaluation"
DEFAULT_GOLD_CSV = ROOT_DIR / "entity_linking" / "1816_el_gs.csv"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Check whether two character-offset spans overlap."""
    return a_start < b_end and b_start < a_end


def _normalise_id(raw: str | None) -> str | None:
    """Normalise a KB ID for comparison.

    Strips ``gn:`` prefix from GeoNames IDs. Returns ``None`` for empty /
    ``"NIL"`` / ``"None"`` values.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.upper() in ("NIL", "NONE", ""):
        return None
    # Strip "gn:" prefix from GeoNames IDs
    if s.startswith("gn:"):
        return s[3:]
    return s


def select_el_file() -> Path:
    """Interactive file picker for ``*_el.spacy`` files in ``el-results/``."""
    files = sorted(EL_RESULTS_DIR.glob("*_el.spacy"))
    if not files:
        print(f"No _el.spacy files found in {EL_RESULTS_DIR}")
        sys.exit(1)
    if len(files) == 1:
        return files[0]
    print("Available EL result files:")
    for i, f in enumerate(files, 1):
        sz = f.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {f.name}  ({sz:.1f} MB)")
    try:
        idx = int(input("Select number: ").strip()) - 1
        if 0 <= idx < len(files):
            return files[idx]
    except (ValueError, EOFError, KeyboardInterrupt):
        pass
    print("Invalid selection.")
    sys.exit(1)


def find_offset_map(el_spacy_path: Path) -> Path | None:
    """Find the matching offset map for an EL .spacy file."""
    stem = el_spacy_path.stem
    # Strip trailing _el suffix to get the base stem
    if stem.endswith("_el"):
        base_stem = stem[:-3]
    else:
        base_stem = stem
    # First try sibling of the _el.spacy file (copied by el.py to el-results/)
    candidate = el_spacy_path.parent / f"{base_stem}_offset_map.json"
    if candidate.exists():
        return candidate
    # Then try flat lookup in EL_RESULTS_DIR
    candidate = EL_RESULTS_DIR / f"{base_stem}_offset_map.json"
    if candidate.exists():
        return candidate
    # Broader fallback in NER output
    for candidate in sorted(NER_RESULTS_DIR.rglob("*offset_map*")):
        if candidate.exists():
            return candidate
    return None


def load_gold_csv(path: Path) -> list[dict]:
    """Load gold standard CSV with entity annotations and KB IDs."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = int(row["start_char"])
                end = int(row["end_char"])
            except (ValueError, KeyError) as e:
                print(f"  Skipping row with invalid offsets: {e}")
                continue
            wd = _normalise_id(row.get("wikidata_qid", ""))
            gn = _normalise_id(row.get("geonames_id", ""))
            rows.append({
                "text": row.get("text", ""),
                "label": row.get("label", ""),
                "start_char": start,
                "end_char": end,
                "wikidata_qid": wd,
                "geonames_id": gn,
                "note": row.get("note", "").strip(),
            })
    return rows


def build_page_boundaries(offset_map: dict) -> list[tuple[int, int, int]]:
    """Return sorted list of ``(page_number, start_offset, end_offset)``."""
    sorted_pages = sorted(offset_map.items(), key=lambda x: int(x[0]))
    boundaries = []
    for i, (page_str, start) in enumerate(sorted_pages):
        page = int(page_str)
        end = (int(sorted_pages[i + 1][1])
               if i + 1 < len(sorted_pages) else float("inf"))
        boundaries.append((page, int(start), end))
    return boundaries


def assign_gold_to_pages(
    gold_rows: list[dict], boundaries: list[tuple[int, int, int]]
) -> dict[int, list[dict]]:
    """Map gold entities to pages by converting doc-level to page-local offsets."""
    gold_by_page: dict[int, list[dict]] = {b[0]: [] for b in boundaries}
    for row in gold_rows:
        for page, page_start, page_end in boundaries:
            if page_start <= row["start_char"] < page_end:
                local = {
                    **row,
                    "local_start": row["start_char"] - page_start,
                    "local_end": row["end_char"] - page_start,
                }
                gold_by_page[page].append(local)
                break
    return gold_by_page


# ── Evaluation ─────────────────────────────────────────────────────────────────

class ELMetrics:
    """Per-KB precision / recall / F1 accumulator."""

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def add(self, tp: int = 0, fp: int = 0, fn: int = 0) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def __repr__(self) -> str:
        return f"TP={self.tp} FP={self.fp} FN={self.fn}  P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f}"


def evaluate(
    gold_by_page: dict[int, list[dict]],
    pred_docs: list,
    boundaries: list[tuple[int, int, int]],
    page_order: list[int],
) -> tuple:
    """Run EL evaluation and return EL-only + full metrics + detail records.

    Returns two levels of metrics:

    **EL-only** (conditioned on span match):
        Only gold entities with a matching pred span are scored.  Unmatched
        gold/pred entities are skipped.  This isolates linking accuracy from
        NER detection errors.

    **Full** (end-to-end):
        All gold entities and all pred entities are scored.  Unmatched gold
        entities count as FN (NER miss), unmatched pred entities count as FP
        (spurious prediction).

    **NIL prediction** (EL-only only):
        Measures how well the model knows when to abstain from linking.
        TP = gold NIL + pred NIL, FP = gold has KB + pred NIL,
        FN = gold NIL + pred has KB, TN = gold has KB + pred has KB.

    Returns:
        ``(el_metrics, full_metrics, details)``

        Each metrics dict has keys ``"wd"``, ``"gn"``, ``"combined"``.
        ``el_metrics`` also has key ``"nil"``.
    """
    el_wd = ELMetrics()
    el_gn = ELMetrics()
    el_combined = ELMetrics()
    el_nil = ELMetrics()

    full_wd = ELMetrics()
    full_gn = ELMetrics()
    full_combined = ELMetrics()

    details: list[dict] = []

    # Map pred docs to pages (docs are in offset-map page order)
    if len(pred_docs) != len(boundaries):
        print(f"  Warning: {len(pred_docs)} docs vs {len(boundaries)} pages")

    doc_by_page: dict[int, int] = {}
    for di in range(min(len(pred_docs), len(page_order))):
        doc_by_page[page_order[di]] = di

    for page, golds in gold_by_page.items():
        if page not in doc_by_page:
            # Page has no pred doc at all — all gold entities are FN in full
            for g in golds:
                g_wd = g["wikidata_qid"]
                g_gn = g["geonames_id"]
                if g_wd:
                    full_wd.add(fn=1)
                if g_gn:
                    full_gn.add(fn=1)
                if g_wd or g_gn:
                    full_combined.add(fn=1)
                details.append({
                    "page": page, "type_combined": "FN",
                    "type_wd": "FN" if g_wd else "",
                    "type_gn": "FN" if g_gn else "",
                    "type_nil_wd": "", "type_nil_gn": "", "type_nil_combined": "",
                    "kb": "unmatched_gold",
                    "gold_text": g["text"], "pred_text": "",
                    "gold_label": g["label"],
                    "gold_wikidata": g_wd or "",
                    "gold_geonames": g_gn or "",
                    "pred_wikidata": "", "pred_geonames": "",
                    "note": g.get("note", ""),
                })
            continue

        doc = pred_docs[doc_by_page[page]]
        preds = list(doc.ents)

        # --- matching pass: find gold-pred pairs by label + span overlap ---
        gold_matched = [False] * len(golds)
        pred_matched = [False] * len(preds)

        for gi, g in enumerate(golds):
            for pi, p in enumerate(preds):
                if (g["label"] == p.label_
                        and _spans_overlap(
                            g["local_start"], g["local_end"],
                            p.start_char, p.end_char)):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    break

        # --- Score each matched pair (EL-only + Full) ---
        for gi, g in enumerate(golds):
            if not gold_matched[gi]:
                # Unmatched gold: FN in full metrics only
                g_wd = g["wikidata_qid"]
                g_gn = g["geonames_id"]
                if g_wd:
                    full_wd.add(fn=1)
                if g_gn:
                    full_gn.add(fn=1)
                if g_wd or g_gn:
                    full_combined.add(fn=1)
                details.append({
                    "page": page, "type_combined": "FN",
                    "type_wd": "FN" if g_wd else "",
                    "type_gn": "FN" if g_gn else "",
                    "type_nil_wd": "", "type_nil_gn": "", "type_nil_combined": "",
                    "kb": "unmatched_gold",
                    "gold_text": g["text"], "pred_text": "",
                    "gold_label": g["label"],
                    "gold_wikidata": g_wd or "",
                    "gold_geonames": g_gn or "",
                    "pred_wikidata": "", "pred_geonames": "",
                    "note": g.get("note", ""),
                })
                continue

            # Find the matching pred
            p = None
            for pi in range(len(preds)):
                if pred_matched[pi] and _spans_overlap(
                        g["local_start"], g["local_end"],
                        preds[pi].start_char, preds[pi].end_char):
                    if g["label"] == preds[pi].label_:
                        p = preds[pi]
                        break
            if p is None:
                continue

            p_wd = _normalise_id(p._.kb_id_wikidata_)
            p_gn = _normalise_id(p._.kb_id_geonames_)
            g_wd = g["wikidata_qid"]
            g_gn = g["geonames_id"]

            p_correct_wd = bool(g_wd and p_wd and p_wd == g_wd)
            p_correct_gn = bool(g_gn and p_gn and p_gn == g_gn)
            is_correct = p_correct_wd or p_correct_gn

            # --- Wikidata ---
            if g_wd:
                if p_correct_wd:
                    el_wd.add(tp=1); full_wd.add(tp=1)
                elif p_wd:
                    el_wd.add(fn=1); el_wd.add(fp=1)
                    full_wd.add(fn=1); full_wd.add(fp=1)
                else:
                    el_wd.add(fn=1); full_wd.add(fn=1)
            else:
                if p_wd:
                    el_wd.add(fp=1); full_wd.add(fp=1)

            # --- GeoNames (independent of Wikidata) ---
            if g_gn:
                if p_correct_gn:
                    el_gn.add(tp=1); full_gn.add(tp=1)
                elif p_gn:
                    el_gn.add(fn=1); el_gn.add(fp=1)
                    full_gn.add(fn=1); full_gn.add(fp=1)
                else:
                    el_gn.add(fn=1); full_gn.add(fn=1)
            else:
                if p_gn:
                    el_gn.add(fp=1); full_gn.add(fp=1)

            # --- Combined: any KB correct? ---
            g_has_any = bool(g_wd or g_gn)
            if g_has_any:
                if is_correct:
                    el_combined.add(tp=1); full_combined.add(tp=1)
                else:
                    el_combined.add(fn=1); full_combined.add(fn=1)
            else:
                if p_wd or p_gn:
                    el_combined.add(fp=1); full_combined.add(fp=1)

            # --- NIL prediction (EL-only only) ---
            # TP = gold NIL + pred NIL, FP = gold has KB + pred NIL,
            # FN = gold NIL + pred has KB, TN = gold has KB + pred has KB
            if not g_wd and not g_gn:
                # Gold is NIL
                if not p_wd and not p_gn:
                    el_nil.add(tp=1)   # correct abstention
                else:
                    el_nil.add(fn=1)   # spurious link
            else:
                # Gold has a KB ID
                if not p_wd and not p_gn:
                    el_nil.add(fp=1)   # missed link (model abstained when it shouldn't)
                # else: TN (correct link) — not counted

            # --- Per-KB type classification ---
            # type_wd
            if g_wd and p_wd and p_wd == g_wd:
                type_wd = "TP"
            elif not g_wd and p_wd:
                type_wd = "FP"
            elif g_wd and (not p_wd or p_wd != g_wd):
                type_wd = "FN"
            else:
                type_wd = "TN"  # neither gold nor pred has Wikidata

            # type_gn
            if g_gn and p_gn and p_gn == g_gn:
                type_gn = "TP"
            elif not g_gn and p_gn:
                type_gn = "FP"
            elif g_gn and (not p_gn or p_gn != g_gn):
                type_gn = "FN"
            else:
                type_gn = "TN"  # neither gold nor pred has GeoNames

            # type_nil_wd — NIL prediction for Wikidata specifically
            if not g_wd and not p_wd:
                type_nil_wd = "TP"  # correct Wikidata abstention
            elif g_wd and not p_wd:
                type_nil_wd = "FP"  # missed Wikidata link
            elif not g_wd and p_wd:
                type_nil_wd = "FN"  # spurious Wikidata link
            else:
                type_nil_wd = "TN"  # both have Wikidata (attempted link)

            # type_nil_gn — NIL prediction for GeoNames specifically
            if not g_gn and not p_gn:
                type_nil_gn = "TP"  # correct GeoNames abstention
            elif g_gn and not p_gn:
                type_nil_gn = "FP"  # missed GeoNames link
            elif not g_gn and p_gn:
                type_nil_gn = "FN"  # spurious GeoNames link
            else:
                type_nil_gn = "TN"  # both have GeoNames (attempted link)

            # type_nil_combined — NIL prediction across both KBs (old type_nil)
            gold_has_kb = bool(g_wd or g_gn)
            pred_has_kb = bool(p_wd or p_gn)
            if not gold_has_kb and not pred_has_kb:
                type_nil_combined = "TP"  # correct abstention
            elif gold_has_kb and not pred_has_kb:
                type_nil_combined = "FP"  # missed link
            elif not gold_has_kb and pred_has_kb:
                type_nil_combined = "FN"  # spurious link
            else:
                type_nil_combined = "TN"  # both have KB (attempted link)

            # type_combined (same semantics as old "type")
            if is_correct:
                type_combined = "TP"
            elif gold_has_kb:
                type_combined = "FP"  # gold has KB but pred wrong/missing
            elif pred_has_kb:
                type_combined = "FP"  # gold NIL but pred spurious
            else:
                type_combined = "TN"  # both NIL

            details.append({
                "page": page, "type_combined": type_combined,
                "type_wd": type_wd, "type_gn": type_gn,
                "type_nil_wd": type_nil_wd, "type_nil_gn": type_nil_gn,
                "type_nil_combined": type_nil_combined,
                "kb": "match",
                "gold_text": g["text"], "pred_text": p.text,
                "gold_label": g["label"],
                "gold_wikidata": g_wd or "",
                "gold_geonames": g_gn or "",
                "pred_wikidata": p_wd or "", "pred_geonames": p_gn or "",
                "note": g.get("note", ""),
            })

        # --- Unmatched pred entities: FP in full metrics ---
        for pi, p in enumerate(preds):
            if not pred_matched[pi]:
                p_wd = _normalise_id(p._.kb_id_wikidata_)
                p_gn = _normalise_id(p._.kb_id_geonames_)
                if p_wd:
                    full_wd.add(fp=1)
                if p_gn:
                    full_gn.add(fp=1)
                if p_wd or p_gn:
                    full_combined.add(fp=1)
                details.append({
                    "page": page, "type_combined": "FP",
                    "type_wd": "FP" if p_wd else "",
                    "type_gn": "FP" if p_gn else "",
                    "type_nil_wd": "", "type_nil_gn": "", "type_nil_combined": "",
                    "kb": "unmatched_pred",
                    "gold_text": "", "pred_text": p.text,
                    "gold_label": p.label_,
                    "gold_wikidata": "", "gold_geonames": "",
                    "pred_wikidata": p_wd or "", "pred_geonames": p_gn or "",
                    "note": "",
                })

    el_metrics = {
        "wd": el_wd, "gn": el_gn, "combined": el_combined, "nil": el_nil,
    }
    full_metrics = {
        "wd": full_wd, "gn": full_gn, "combined": full_combined,
    }
    return el_metrics, full_metrics, details


# ── Reporting ──────────────────────────────────────────────────────────────────

def _print_metrics_header(level_name: str, description: str) -> None:
    """Print a section header with a clear explanation of what the metrics mean."""
    print()
    print("=" * 70)
    print(f"  {level_name}")
    print(f"  {description}")
    print("=" * 70)


def _print_metrics_table(
    wd: ELMetrics, gn: ELMetrics, combined: ELMetrics,
    nil: ELMetrics | None = None,
) -> None:
    """Print a formatted metrics table with TP/FP/FN and P/R/F1."""
    print(f"    {'KB':<20} {'TP':>5} {'FP':>5} {'FN':>5}  "
          f"{'Precision':>8} {'Recall':>8} {'F1':>8}  {'Explanation':<30}")
    print("    " + "-" * 100)
    for kb_name, m in [("Wikidata", wd), ("GeoNames", gn), ("Combined", combined)]:
        if kb_name == "Combined":
            expl = "correct in either KB"
        elif kb_name == "Wikidata":
            expl = "QID match vs gold QID"
        else:
            expl = "GN-ID match vs gold GN-ID"
        print(f"    {kb_name:<20} {m.tp:>5} {m.fp:>5} {m.fn:>5}  "
              f"{m.precision:>8.3f} {m.recall:>8.3f} {m.f1:>8.3f}  {expl:<30}")
    if nil is not None:
        print(f"    {'NIL (abstention)':<20} {nil.tp:>5} {nil.fp:>5} {nil.fn:>5}  "
              f"{nil.precision:>8.3f} {nil.recall:>8.3f} {nil.f1:>8.3f}  "
              f"{'correct NIL vs spurious/missed':<30}")


def _print_metrics(
    label: str, wd: ELMetrics, gn: ELMetrics, combined: ELMetrics,
    nil: ELMetrics | None = None,
) -> None:
    print(f"\n  {label}:")
    print(f"    {'':<20} {'TP':>5} {'FP':>5} {'FN':>5}  {'P':>8} {'R':>8} {'F1':>8}")
    sep = "    " + "-" * 60
    print(sep)
    for kb_name, m in [("Wikidata", wd), ("GeoNames", gn), ("Combined", combined)]:
        print(f"    {kb_name:<20} {m.tp:>5} {m.fp:>5} {m.fn:>5}  "
              f"{m.precision:>8.3f} {m.recall:>8.3f} {m.f1:>8.3f}")
    if nil is not None:
        print(f"    {'NIL (abstention)':<20} {nil.tp:>5} {nil.fp:>5} {nil.fn:>5}  "
              f"{nil.precision:>8.3f} {nil.recall:>8.3f} {nil.f1:>8.3f}")


def _print_per_label(per_label: dict[str, dict]) -> None:
    if not per_label:
        return
    print("\n  Per-Label Breakdown (Wikidata P / R / F1 — EL-only):")
    print(f"    {'Label':<30} {'P':>8} {'R':>8} {'F1':>8}")
    print("    " + "-" * 56)
    for lbl in sorted(per_label):
        m = per_label[lbl]["wd"]
        print(f"    {lbl:<30} {m.precision:>8.3f} {m.recall:>8.3f} {m.f1:>8.3f}")
    print("\n  Per-Label Breakdown (GeoNames P / R / F1 — EL-only):")
    print(f"    {'Label':<30} {'P':>8} {'R':>8} {'F1':>8}")
    print("    " + "-" * 56)
    for lbl in sorted(per_label):
        m = per_label[lbl]["gn"]
        print(f"    {lbl:<30} {m.precision:>8.3f} {m.recall:>8.3f} {m.f1:>8.3f}")
    print("\n  Per-Label Breakdown (NIL P / R / F1 — EL-only):")
    print(f"    {'Label':<30} {'P':>8} {'R':>8} {'F1':>8}")
    print("    " + "-" * 56)
    for lbl in sorted(per_label):
        m = per_label[lbl].get("nil")
        if m is not None:
            print(f"    {lbl:<30} {m.precision:>8.3f} {m.recall:>8.3f} {m.f1:>8.3f}")


def _write_detail_csv(details: list[dict], path: Path) -> None:
    """Write per-instance evaluation records to a CSV."""
    fieldnames = [
        "page", "type_combined", "type_wd", "type_gn",
        "type_nil_wd", "type_nil_gn", "type_nil_combined", "kb",
        "gold_text", "pred_text", "gold_label",
        "gold_wikidata", "gold_geonames",
        "pred_wikidata", "pred_geonames", "note",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)
    print(f"\n  Detail CSV → {path.name}  ({len(details)} records)")


def _append_scores_csv(
    path: Path,
    source_text: str,
    model: str,
    temperature: float,
    top_k_rerank: int,
    inference_type: str,
    total_entities: int,
    duration_seconds: float,
    pipeline_stats: dict,
    wd: ELMetrics,
    gn: ELMetrics,
    combined: ELMetrics,
    nil: ELMetrics | None = None,
    think: str = "",
) -> None:
    """Append one summary row to a cumulative EL scores CSV.

    Creates the file with a header row if it doesn't exist yet; otherwise
    appends a data row, so multiple evaluation runs accumulate into one file.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fieldnames = [
        "source_text", "model", "temperature", "top_k_rerank",
        "think",
        "inference_type",
        "total_entities", "duration_seconds",
        "datetime",
        "count_errors", "count_skipped",
        "count_linked_wd", "count_linked_gn",
        "count_linked_both", "count_linked_neither",
        "wd_p", "wd_r", "wd_f1",
        "gn_p", "gn_r", "gn_f1",
        "combined_p", "combined_r", "combined_f1",
        "nil_p", "nil_r", "nil_f1",
    ]
    file_exists = path.exists()
    needs_header = not file_exists

    if file_exists and path.stat().st_size > 0:
        # Check if the existing file already has a header row
        with open(path, encoding="utf-8") as f:
            first_line = f.readline().strip()
        expected_header = ",".join(fieldnames)
        if first_line != expected_header:
            # Old-format file without header — recreate with header
            old_rows = []
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f, fieldnames=fieldnames)
                for row in reader:
                    # Only keep rows that look like data (not the old header)
                    if row.get("source_text", "") != first_line.split(",")[0]:
                        old_rows.append(row)
            # Rewrite with header + old data
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(old_rows)
            needs_header = False  # header already written

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()

        row = {
            "source_text": source_text,
            "model": model,
            "temperature": temperature,
            "top_k_rerank": top_k_rerank,
            "think": think,
            "inference_type": inference_type,
            "total_entities": total_entities,
            "duration_seconds": f"{duration_seconds:.1f}",
            "datetime": now,
            "count_errors": pipeline_stats.get("count_errors", 0),
            "count_skipped": pipeline_stats.get("count_skipped", 0),
            "count_linked_wd": pipeline_stats.get("count_linked_wd", 0),
            "count_linked_gn": pipeline_stats.get("count_linked_gn", 0),
            "count_linked_both": pipeline_stats.get("count_linked_both", 0),
            "count_linked_neither": pipeline_stats.get("count_linked_neither", 0),
            "wd_p": f"{wd.precision:.4f}",
            "wd_r": f"{wd.recall:.4f}",
            "wd_f1": f"{wd.f1:.4f}",
            "gn_p": f"{gn.precision:.4f}",
            "gn_r": f"{gn.recall:.4f}",
            "gn_f1": f"{gn.f1:.4f}",
            "combined_p": f"{combined.precision:.4f}",
            "combined_r": f"{combined.recall:.4f}",
            "combined_f1": f"{combined.f1:.4f}",
            "nil_p": f"{nil.precision:.4f}" if nil else "",
            "nil_r": f"{nil.recall:.4f}" if nil else "",
            "nil_f1": f"{nil.f1:.4f}" if nil else "",
        }
        writer.writerow(row)


# ── Main ───────────────────────────────────────────────────────────────────────

def _parse_model_from_stem(stem: str) -> str:
    """Extract just the EL model name from a filename stem.

    Handles naming patterns::

        {source}__{el_model}_t{temp}[_think{mode}]
        {source}__{ner_model}_t{temp}_{ner_config}__{el_model}_t{temp2}[_think{mode}]

    Returns the model name only (no temperature, no think suffix).
    """
    if stem.endswith("_el"):
        stem = stem[:-3]
    # Take the last __-delimited segment — that holds the EL model info
    if "__" in stem:
        last = stem.split("__")[-1]
    else:
        last = stem
    # Strip _t{temp} and anything after
    m = re.search(r"_t\d+\.?\d*", last)
    if m:
        model_part = last[:m.start()]
    else:
        model_part = last
    # Restore colon for model family:version patterns
    if "-" in model_part:
        parts = model_part.rsplit("-", 1)
        if parts[1] and parts[1][0].isdigit():
            model_part = f"{parts[0]}:{parts[1]}"
    return model_part


def _find_run_info(el_spacy_path: Path) -> Path | None:
    """Find the companion ``*_run_info.json`` for an ``*_el.spacy`` file."""
    stem = el_spacy_path.stem
    if stem.endswith("_el"):
        base = stem[:-3]
    else:
        base = stem
    candidate = el_spacy_path.parent / f"{base}_run_info.json"
    if candidate.exists():
        return candidate
    candidates = sorted(el_spacy_path.parent.glob(f"{base}*run_info*"))
    if candidates:
        return candidates[0]
    return None


def _parse_filename_metadata(stem: str) -> dict:
    """Parse model, temperature, and think mode from a filename stem.

    Handles naming patterns::

        {source}__{model}_t{temp}[_think{mode}]
        {source}__{ner_model}_t{temp}_{ner_config}__{el_model}_t{temp2}[_think{mode}]

    In filenames, ``:`` in model names is replaced with ``-`` (OS-safe),
    so ``gemma4:31b`` becomes ``gemma4-31b``.  We convert the most likely
    ``{family}-{version}`` pattern back to ``{family}:{version}``.
    """
    if stem.endswith("_el"):
        stem = stem[:-3]

    if "__" in stem:
        last = stem.split("__")[-1]
    else:
        last = stem

    m = re.search(r"_t(\d+\.?\d*)", last)
    if m:
        temperature = float(m.group(1))
        model_part = last[:m.start()]
        after_temp = last[m.end():]
        think = ""
        think_m = re.search(r"_think(\w+)", after_temp)
        if think_m:
            think = think_m.group(1)
    else:
        temperature = 0.1
        model_part = last
        think = ""

    if "-" in model_part:
        parts = model_part.rsplit("-", 1)
        if parts[1] and parts[1][0].isdigit():
            model_part = f"{parts[0]}:{parts[1]}"

    return {"model": model_part, "temperature": temperature, "think": think}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate EL predictions against gold-standard KB IDs."
    )
    parser.add_argument("--gold", default=None,
                        help="Path to gold CSV with wikidata_qid / geonames_id")
    parser.add_argument("--el", default=None,
                        help="Path to EL-predicted _el.spacy file")
    parser.add_argument("--detail", action="store_true", default=True,
                        help="Write per-instance detail CSV (default: True)")
    parser.add_argument("--pipeline-stats", default=None,
                        help="JSON string of pipeline stats from link_entities()")
    parser.add_argument("--duration", type=float, default=None,
                        help="EL pipeline duration in seconds")
    parser.add_argument("--model", default=None,
                        help="Model name used for EL pipeline")
    parser.add_argument("--temperature", type=float, default=None,
                        help="LLM temperature used for EL pipeline")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k rerank value used for EL pipeline")
    parser.add_argument("--think", type=str, default=None,
                        help="Thinking mode used for EL pipeline (false, low, medium, high)")
    parser.add_argument("--inference-type", default=None,
                        help="'cloud' or 'local' — overrides auto-detection from model name")
    parser.add_argument("--source-text", default="",
                        help="Source text label for scores.csv (default: gold CSV stem)")
    args = parser.parse_args()

    # ── 0. Auto-discover metadata from companion _run_info.json or filename ──
    el_path = Path(args.el) if args.el else select_el_file()
    if not el_path.exists():
        print(f"EL file not found: {el_path}")
        sys.exit(1)

    run_info = None
    run_info_path = _find_run_info(el_path)
    if run_info_path:
        try:
            with open(run_info_path) as f:
                run_info = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    if run_info:
        auto_model = run_info.get("model", "")
        auto_temperature = run_info.get("temperature")
        auto_think = run_info.get("think", "")
        auto_top_k = run_info.get("top_k", 3)
        auto_duration = run_info.get("duration", 0.0)
        auto_stats = run_info.get("stats", {})
        auto_inference = run_info.get("inference_type", "")
    else:
        filename_meta = _parse_filename_metadata(el_path.stem)
        auto_model = filename_meta.get("model", "")
        auto_temperature = filename_meta.get("temperature")
        auto_think = filename_meta.get("think", "")
        auto_top_k = 3
        auto_duration = 0.0
        auto_stats = {}
        auto_inference = ""

    # CLI args override auto-discovered values
    model_name = args.model or auto_model
    temperature = args.temperature if args.temperature is not None else (auto_temperature if auto_temperature is not None else 0.1)
    think = args.think if args.think is not None else auto_think
    top_k = args.top_k if args.top_k is not None else auto_top_k
    duration = args.duration if args.duration is not None else auto_duration
    pipeline_stats = auto_stats
    if args.pipeline_stats:
        try:
            pipeline_stats = json.loads(args.pipeline_stats)
        except (json.JSONDecodeError, TypeError):
            print("  Warning: could not parse --pipeline-stats JSON")
    if args.inference_type:
        inference_type = args.inference_type
    elif auto_inference:
        inference_type = auto_inference
    else:
        inference_type = "cloud" if "cloud" in model_name else "local"
        

    # ── 1. Load files ──────────────────────────────────────────────────────────
    gold_path = Path(args.gold) if args.gold else DEFAULT_GOLD_CSV
    if not gold_path.exists():
        print(f"Gold CSV not found: {gold_path}")
        sys.exit(1)
    
    source_name = args.source_text or gold_path.stem

    offset_map_path = find_offset_map(el_path)
    if not offset_map_path or not offset_map_path.exists():
        print(f"No offset map found for {el_path.name}")
        sys.exit(1)

    print(f"Gold CSV:  {gold_path.name}")
    print(f"EL file:   {el_path.name}")
    print(f"Offset:    {offset_map_path.name}")
    print()

    gold_rows = load_gold_csv(gold_path)
    print(f"Gold entities loaded: {len(gold_rows)}")
    n_gold_has_wd = sum(1 for r in gold_rows if r["wikidata_qid"] is not None)
    n_gold_has_gn = sum(1 for r in gold_rows if r["geonames_id"] is not None)
    n_gold_has_any = sum(
        1 for r in gold_rows
        if r["wikidata_qid"] is not None or r["geonames_id"] is not None
    )

    with open(offset_map_path, encoding="utf-8") as f:
        offset_map = json.load(f)
    boundaries = build_page_boundaries(offset_map)

    nlp = spacy.blank("nl")
    db = DocBin().from_disk(str(el_path))
    pred_docs = list(db.get_docs(nlp.vocab))
    print(f"Predicted docs loaded: {len(pred_docs)}")

    # Page order (sorted by offset)
    page_order = [b[0] for b in boundaries]

    # ── 2. Assign gold to pages ────────────────────────────────────────────────
    gold_by_page = assign_gold_to_pages(gold_rows, boundaries)
    total_assigned = sum(len(v) for v in gold_by_page.values())
    print(f"Gold assigned to pages: {total_assigned}")

    # ── 3. Run evaluation ──────────────────────────────────────────────────────
    el_metrics, full_metrics, details = evaluate(
        gold_by_page, pred_docs, boundaries, page_order,
    )

    # ── 4. Report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ENTITY LINKING EVALUATION RESULTS")
    print("=" * 70)

    # ── 4a. Explanation of TP / FP / FN ──────────────────────────────────────
    print()
    print("  HOW TP / FP / FN ARE COUNTED (per KB — Wikidata and GeoNames separately):")
    print()
    print("    For each gold entity that has a KB ID (e.g. a Wikidata QID):")
    print("      -> TP if the predicted KB ID matches the gold KB ID exactly")
    print("      -> FN if the predicted KB ID is wrong OR missing (we failed to link)")
    print("      -> FP if the predicted KB ID is wrong (a wrong link was made)")
    print("         Note: when gold has QID=X and pred has QID=Y (wrong),")
    print("         that counts as 1 FN (missed X) + 1 FP (wrongly gave Y)")
    print()
    print("    For each gold entity that has NO KB ID (blank/NIL in gold):")
    print("      -> FP if the predictor still assigns a KB ID (spurious link)")
    print("      -> nothing if the predictor also assigns nothing (correct abstain)")
    print()
    print("    'Combined' = correct in EITHER Wikidata OR GeoNames (or both).")
    print("    This is the most lenient measure: a link counts as correct if at")
    print("    least one of the two KBs has the right ID.")
    print()
    print("    'NIL (abstention)' = measures how well the model knows when to abstain.")
    print("      TP = gold NIL + pred NIL (correct abstention)")
    print("      FP = gold has KB + pred NIL (model missed a link)")
    print("      FN = gold NIL + pred has KB (model linked when it should not have)")

    # ── 4b. EL-only metrics table ─────────────────────────────────────────────
    _print_metrics_header(
        "EL-ONLY METRICS (conditioned on span match)",
        "Only gold entities with a matching predicted span are scored.\n"
        "  Unmatched gold/pred entities are skipped.\n"
        "  This isolates linking accuracy from NER detection errors."
    )
    _print_metrics_table(
        el_metrics["wd"], el_metrics["gn"], el_metrics["combined"],
        nil=el_metrics["nil"],
    )

    # ── 4c. Full metrics table ────────────────────────────────────────────────
    _print_metrics_header(
        "FULL METRICS (end-to-end, includes NER errors)",
        "All gold entities and all predicted entities are scored.\n"
        "  Unmatched gold entities count as FN (NER miss).\n"
        "  Unmatched pred entities count as FP (spurious prediction).\n"
        "  NIL metrics are not computed at the full level (NIL is inherently EL-only)."
    )
    _print_metrics_table(
        full_metrics["wd"], full_metrics["gn"], full_metrics["combined"],
    )

    # ── 4d. Per-label breakdown ─────────────────────────────────────────────
    per_label = _build_per_label(gold_by_page, pred_docs, boundaries, page_order)
    _print_per_label(per_label)

    # ── 4e. Summary counts ──────────────────────────────────────────────────
    print()
    print("-" * 70)
    print("  SUMMARY - Gold Standard Composition")
    print("-" * 70)
    print(f"  Total gold entities:              {len(gold_rows)}")
    print(f"    with Wikidata QID:              {n_gold_has_wd}  "
          f"({n_gold_has_wd/len(gold_rows)*100:.1f}%)")
    print(f"    with GeoNames ID:               {n_gold_has_gn}  "
          f"({n_gold_has_gn/len(gold_rows)*100:.1f}%)")
    print(f"    with any KB ID:                 {n_gold_has_any}  "
          f"({n_gold_has_any/len(gold_rows)*100:.1f}%)")
    print(f"    blank (no link in gold):        {len(gold_rows) - n_gold_has_any}  "
          f"({(len(gold_rows)-n_gold_has_any)/len(gold_rows)*100:.1f}%)")

    # ── 5. Detail CSV ──────────────────────────────────────────────────────────
    if args.detail:
        detail_path = el_path.with_name(el_path.stem + "_el_eval.csv")
        _write_detail_csv(details, detail_path)

    # ── 6. Scores CSV ──────────────────────────────────────────────────────────
    EL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = EL_EVAL_DIR / "scores.csv"
    _append_scores_csv(
        path=scores_path,
        source_text=source_name,
        model=model_name,
        temperature=temperature,
        top_k_rerank=top_k,
        inference_type=inference_type,
        total_entities=len(gold_rows),
        duration_seconds=duration,
        pipeline_stats=pipeline_stats,
        wd=full_metrics["wd"], gn=full_metrics["gn"], combined=full_metrics["combined"],
        nil=el_metrics["nil"],
        think=think,
    )
    print(f"\n  Scores CSV → {scores_path.name}")


def _build_per_label(
    gold_by_page: dict[int, list[dict]],
    pred_docs: list,
    boundaries: list[tuple[int, int, int]],
    page_order: list[int],
) -> dict[str, dict]:
    """Build per-label metric accumulators separately for the report."""
    per_label: dict[str, dict] = {}

    doc_by_page: dict[int, int] = {}
    for di in range(min(len(pred_docs), len(page_order))):
        doc_by_page[page_order[di]] = di

    for page, golds in gold_by_page.items():
        if page not in doc_by_page:
            continue
        doc = pred_docs[doc_by_page[page]]
        preds = list(doc.ents)

        for g in golds:
            lbl = g["label"]
            if lbl not in per_label:
                per_label[lbl] = {
                    "wd": ELMetrics(), "gn": ELMetrics(), "nil": ELMetrics(),
                }

            # Find matching pred
            p = None
            for pe in preds:
                if (g["label"] == pe.label_
                        and _spans_overlap(
                            g["local_start"], g["local_end"],
                            pe.start_char, pe.end_char)):
                    p = pe
                    break

            if p is None:
                continue

            p_wd = _normalise_id(p._.kb_id_wikidata_)
            p_gn = _normalise_id(p._.kb_id_geonames_)
            g_wd = g["wikidata_qid"]
            g_gn = g["geonames_id"]

            pl = per_label[lbl]
            if g_wd:
                if p_wd and p_wd == g_wd:
                    pl["wd"].add(tp=1)
                else:
                    pl["wd"].add(fn=1)
            else:
                if p_wd:
                    pl["wd"].add(fp=1)
            if g_gn:
                if p_gn and p_gn == g_gn:
                    pl["gn"].add(tp=1)
                else:
                    pl["gn"].add(fn=1)
            else:
                if p_gn:
                    pl["gn"].add(fp=1)

            # NIL per label
            if not g_wd and not g_gn:
                if not p_wd and not p_gn:
                    pl["nil"].add(tp=1)
                else:
                    pl["nil"].add(fn=1)
            else:
                if not p_wd and not p_gn:
                    pl["nil"].add(fp=1)

    return per_label


if __name__ == "__main__":
    main()
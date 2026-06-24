"""Evaluate NER predictions against Recogito manual annotations.

Loads a predicted .spacy DocBin and a Recogito JSON-LD export, aligns them
via full-text character offsets, and reports precision/recall/F1 per label.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from spacy.scorer import Scorer
from spacy.training import Example
from spacy.tokens import DocBin
import spacy

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "ner-output"
EVAL_DIR = SCRIPT_DIR / "ner-evaluation"
ERRORS_DIR = EVAL_DIR / "errors_logs"


def get_annotations_path(spacy_file: Path) -> tuple[Path, str, Path]:
    """Derive the GS annotations file, source text prefix, and source text path.

    Parses the spacy filename (e.g. '1816_third_letter__...') and returns
    the matching '{prefix}-gs-annotations.jsonld' file, the prefix, and
    the expected source .txt path (relative to repo root).

    Returns (annotations_path, prefix, source_text_path).
    """
    stem = spacy_file.stem
    try:
        source_text_label, _ = stem.split("__", maxsplit=1)
        prefix = source_text_label.split("_", maxsplit=1)[0]  # e.g. '1816' or '1809'
    except (ValueError, IndexError):
        prefix = ""
        source_text_label = ""
    annotations_path = SCRIPT_DIR / "gs_annotations" / f"{prefix}.jsonld"
    if not annotations_path.exists():
        print(f"Error: no GS annotations found for prefix '{prefix}'")
        print(f"  Tried: {annotations_path}")
        sys.exit(1)
    source_text_path = SCRIPT_DIR.parent / "data" / f"{source_text_label}.txt"
    return annotations_path, prefix, source_text_path


TAG_MAPS = {
    "1816": {
        "E53 Place": "E53_Place",
        "E18 Physical Thing": "E18_Physical_Thing",
        "E22 Human-made Object": "Mode_of_Transportation",
        "Mode of Transportation": "Mode_of_Transportation",
        "Mode_of_Transportation": "Mode_of_Transportation",
        "F2 Expression": "F2_Expression",
        "E52 Time-Span": "E52_Time_Span",
        "E19 Physical Object": "E19_Physical_Object",
        "E20 Biological Object": "E20_Biological_Object",
        "E31 Document": "E31_Document",
    },
    "1809": {
        "E53 Place": "E53_Place",
        "E18 Physical Thing": "E18_Physical_Thing",
        "Mode of Transportation": "Mode_of_Transportation",
        "Mode_of_Transportation": "Mode_of_Transportation",
        "E52 Time-Span": "E52_Time_Span",
        "F2 Expression": "F2_Expression",
        "E19 Physical Object": "E19_Physical_Object",
        "E20 Biological Object": "E20_Biological_Object",
        "E31 Document": "E31_Document",
    },
}


def select_results_file() -> Path:
    """Present a numbered menu of .spacy files in ner-output/ and return the chosen path."""
    spacy_files = sorted(RESULTS_DIR.rglob("*.spacy"))
    if not spacy_files:
        print("No .spacy files found in ner-output/")
        sys.exit(1)
    print("Available .spacy files:", flush=True)
    for i, f in enumerate(spacy_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        rel = f.relative_to(RESULTS_DIR)
        print(f"  {i:>3}. {rel}  ({size_mb:.1f} MB)", flush=True)
    try:
        choice = input(f"\nSelect a file (1-{len(spacy_files)}): ").strip()
        idx = int(choice) - 1
        if idx < 0 or idx >= len(spacy_files):
            raise ValueError
    except (ValueError, EOFError, KeyboardInterrupt):
        print("Invalid selection.")
        sys.exit(1)
    return spacy_files[idx]


def load_recogito_annotations(
    path: str, tag_map: dict[str, str], source_path: Path,
    offset_map: dict | None = None,
) -> list[dict]:
    """Extract annotations with purpose=tagging from Recogito JSON-LD.

    Recogito stores character offsets against the raw source file opened in
    Python text mode (UTF-8 BOM present, CRLF→LF). The NER pipeline normalizes
    the same text through BOM strip + chunk split + .strip().

    This function builds a precise per-character offset remap from raw text
    to the concatenated stripped text that the pipeline produces, accounting
    for BOM removal, CRLF→LF conversion, delimiter removal, and chunk stripping.
    """
    import re

    with open(path) as f:
        data = json.load(f)

    # Read source in text mode — this is Recogito's coordinate space
    with open(source_path) as f:
        raw_text = f.read()

    # Strip BOM to match pipeline coordinate space
    bom_len = 1 if raw_text.startswith("﻿") else 0
    raw_text = raw_text[bom_len:]

    if offset_map is None:
        print(f"Loaded 0 annotations from {path} (no offset map provided)")
        return []

    # --- Build precise offset remap: raw_text position → concat_stripped position ---
    # Step 1: Find all delimiter positions (same regex as ner.py)
    delimiters = [(m.start(), m.end()) for m in re.finditer(r"\n_{16,}\n", raw_text)]

    # Step 2: Compute chunk positions (gaps between delimiters)
    chunk_spans = []
    prev_end = 0
    for d_start, d_end in delimiters:
        chunk_spans.append((prev_end, d_start))
        prev_end = d_end
    chunk_spans.append((prev_end, len(raw_text)))

    # Step 3: Build stripped chunks + their concat positions (same logic as pipeline)
    stripped_info = []  # list of (raw_start, raw_end, stripped_text, concat_start)
    concat_pos = 0
    for cs, ce in chunk_spans:
        chunk_text = raw_text[cs:ce]
        stripped = chunk_text.strip()
        if stripped:
            # Count leading whitespace stripped
            leading = len(chunk_text) - len(chunk_text.lstrip())
            stripped_info.append((cs, ce, stripped, concat_pos, leading))
            concat_pos += len(stripped)

    # Step 4: Build a raw→concat position map for quick lookup
    # For each position in raw_text, compute its concat position
    # Positions in delimiters or in stripped whitespace → map to nearest valid position
    raw_to_concat = {}
    for cs, ce, stripped, concat_start, leading in stripped_info:
        # Positions in leading whitespace → map to concat_start
        for p in range(cs, cs + leading):
            raw_to_concat[p] = concat_start
        # Positions in the actual stripped content
        for i in range(len(stripped)):
            raw_to_concat[cs + leading + i] = concat_start + i
        # Positions in trailing whitespace → map to concat_start + len(stripped)
        trailing_start = cs + leading + len(stripped)
        for p in range(trailing_start, ce):
            raw_to_concat[p] = concat_start + len(stripped)
    # Positions in delimiters → map to the concat position of the next chunk
    for d_start, d_end in delimiters:
        for p in range(d_start, d_end):
            # Find next non-delimiter position
            next_pos = d_end
            raw_to_concat[p] = raw_to_concat.get(next_pos, concat_pos)

    # Step 5: Remap all annotations using the precise offset map
    annotations = []
    missed = 0
    for item in data:
        bodies = item.get("body", [])
        if not isinstance(bodies, list):
            bodies = [bodies]
        tag = None
        for b in bodies:
            if b.get("purpose") == "tagging":
                tag = b.get("value", "")
                break
        if not tag:
            continue
        tag = tag_map.get(tag, tag)
        selector = item.get("target", {}).get("selector", [])
        pos = None
        for sel in selector:
            if sel.get("type") == "TextPositionSelector":
                pos = (sel["start"], sel["end"])
                break
        if pos is None:
            continue

        # Adjust for BOM
        raw_start = pos[0] - bom_len
        raw_end = pos[1] - bom_len

        # Remap using the precise offset map
        concat_start = raw_to_concat.get(raw_start)
        concat_end = raw_to_concat.get(raw_end)

        if concat_start is not None and concat_end is not None and concat_end > concat_start:
            annotations.append({
                "start": concat_start,
                "end": concat_end,
                "label": tag,
            })
        else:
            missed += 1

    if missed:
        print(f"  Warning: {missed}/{len(data)} annotations could not be aligned")

    print(f"Loaded {len(annotations)} annotations from {path}")
    return annotations


def build_page_boundaries(offset_map: dict) -> list[tuple]:
    """Return sorted (page_number, start_offset, end_offset) list."""
    sorted_pages = sorted(offset_map.items(), key=lambda x: int(x[0]))
    boundaries = []
    for i, (page_str, start) in enumerate(sorted_pages):
        page = int(page_str)
        end = int(sorted_pages[i + 1][1]) if i + 1 < len(sorted_pages) else float("inf")
        boundaries.append((page, int(start), end))
    return boundaries


def assign_annotations_to_pages(
    annotations: list[dict], boundaries: list[tuple]
) -> dict:
    """Map annotations to page numbers based on full-text offsets."""
    page_anns: dict = {b[0]: [] for b in boundaries}
    for ann in annotations:
        for page, start, end in boundaries:
            if start <= ann["start"] < end:
                local_start = ann["start"] - start
                local_end = ann["end"] - start
                page_anns[page].append(
                    {"start": local_start, "end": local_end, "label": ann["label"]}
                )
                break
    return page_anns


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Check whether two character-offset spans overlap."""
    return a_start < b_end and b_start < a_end


def compute_relaxed_scores(
    gold_spans: list[tuple[int, int, str]],
    pred_spans: list[tuple[int, int, str]],
) -> dict:
    """Compute P/R/F1 where any span overlap with same label counts as a TP."""
    if not gold_spans and not pred_spans:
        return {"p": 1.0, "r": 1.0, "f": 1.0}

    gold_matched = [False] * len(gold_spans)
    pred_matched = [False] * len(pred_spans)

    for gi, (gs, ge, gl) in enumerate(gold_spans):
        for pi, (ps, pe, pl) in enumerate(pred_spans):
            if gl == pl and _spans_overlap(gs, ge, ps, pe):
                gold_matched[gi] = True
                pred_matched[pi] = True

    tp = sum(gold_matched)
    fp = len(pred_spans) - sum(pred_matched)
    fn = len(gold_spans) - tp

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"p": p, "r": r, "f": f}


def compute_relaxed_per_label(
    gold_spans: list[tuple[int, int, str]],
    pred_spans: list[tuple[int, int, str]],
    labels: set[str],
) -> dict:
    """Compute relaxed P/R/F1 per label."""
    result = {}
    for label in labels:
        g = [(s, e, l) for s, e, l in gold_spans if l == label]
        p = [(s, e, l) for s, e, l in pred_spans if l == label]
        result[label] = compute_relaxed_scores(g, p)
    return result


def classify_instances(
    pages_gold: list[list[tuple[int, int, str, str]]],
    pages_pred: list[list[tuple[int, int, str, str]]],
    page_numbers: list[int],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Classify each gold and predicted span as TP, FP, or FN via relaxed overlap matching.

    Each span tuple is (start_char, end_char, label, text). A gold span and a
    pred span count as a true positive when they share a label and their character
    offsets overlap. Returns three lists of dicts ready for CSV export.
    """
    tp_instances = []
    fp_instances = []
    fn_instances = []

    for page_gold, page_pred, page_num in zip(pages_gold, pages_pred, page_numbers):
        gold_matched = [False] * len(page_gold)
        pred_matched = [False] * len(page_pred)

        for gi, (gs, ge, gl, _gt) in enumerate(page_gold):
            for pi, (ps, pe, pl, _pt) in enumerate(page_pred):
                if gl == pl and _spans_overlap(gs, ge, ps, pe):
                    gold_matched[gi] = True
                    pred_matched[pi] = True

        for gi, (gs, ge, gl, gt) in enumerate(page_gold):
            if gold_matched[gi]:
                for pi, (ps, pe, pl, pt) in enumerate(page_pred):
                    if pl == gl and _spans_overlap(gs, ge, ps, pe):
                        tp_instances.append({
                            "type": "TP", "label": gl, "page": page_num,
                            "gold_text": gt, "pred_text": pt,
                            "gold_start": gs, "gold_end": ge,
                            "pred_start": ps, "pred_end": pe,
                        })
                        break
            else:
                fn_instances.append({
                    "type": "FN", "label": gl, "page": page_num,
                    "gold_text": gt, "pred_text": "",
                    "gold_start": gs, "gold_end": ge,
                    "pred_start": "", "pred_end": "",
                })

        for pi, (ps, pe, pl, pt) in enumerate(page_pred):
            if not pred_matched[pi]:
                fp_instances.append({
                    "type": "FP", "label": pl, "page": page_num,
                    "gold_text": "", "pred_text": pt,
                    "gold_start": "", "gold_end": "",
                    "pred_start": ps, "pred_end": pe,
                })

    return tp_instances, fp_instances, fn_instances


def write_errors_csv(
    tp_instances: list[dict],
    fp_instances: list[dict],
    fn_instances: list[dict],
    output_path: Path,
):
    """Write TP/FP/FN instances to a CSV file sorted by page number then error type."""
    fieldnames = [
        "type", "label", "page",
        "gold_text", "pred_text",
        "gold_start", "gold_end", "pred_start", "pred_end",
    ]
    type_order = {"FN": 0, "FP": 1, "TP": 2}
    all_instances = tp_instances + fp_instances + fn_instances
    all_instances.sort(key=lambda x: (
        x["page"],
        type_order.get(x["type"], 9),
        x["gold_start"] if x["gold_start"] != "" else x["pred_start"],
    ))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_instances)

    print(f"\nErrors CSV written to {output_path}")
    print(f"  TP: {len(tp_instances)}, FP: {len(fp_instances)}, FN: {len(fn_instances)}")


def parse_args():
    """Parse CLI arguments for non-interactive usage."""
    parser = argparse.ArgumentParser(
        description="Evaluate NER predictions against Recogito manual annotations."
    )
    parser.add_argument(
        "-s", "--spacy-file", type=str, default=None,
        help="Path to a .spacy file to evaluate. If omitted, shows an interactive file picker."
    )
    return parser.parse_args()


def main():
    """Evaluate NER predictions against Recogito gold annotations.

    Matches predicted docs to pages via the full-text offset map, builds spaCy
    Example objects for strict scoring, and computes relaxed (overlap-based)
    P/R/F1 per label. Exports TP/FP/FN instances to a CSV.

    If --spacy-file is provided on the CLI, evaluate that file directly.
    Otherwise, show an interactive file picker.
    """
    # 1. Select files — CLI arg or interactive picker
    args = parse_args()
    if args.spacy_file:
        spacy_file = Path(args.spacy_file).resolve()
        if not spacy_file.exists():
            print(f"Error: spacy file not found: {spacy_file}")
            sys.exit(1)
        if spacy_file.suffix != ".spacy":
            print(f"Error: expected a .spacy file, got: {spacy_file.suffix}")
            sys.exit(1)
    else:
        spacy_file = select_results_file()

    offset_map_path = spacy_file.with_name(
        spacy_file.stem + "_offset_map.json"
    )
    if not offset_map_path.exists():
        print(f"No offset map found: {offset_map_path}")
        print("Re-run ner.py to generate one (markers are now kept in chunk text).")
        sys.exit(1)

    recogito_file, prefix, source_text_path = get_annotations_path(spacy_file)

    # 2. Load data
    with open(offset_map_path) as f:
        offset_map = json.load(f)

    tag_map = TAG_MAPS.get(prefix, TAG_MAPS.get("1816", {}))
    annotations = load_recogito_annotations(str(recogito_file), tag_map, source_text_path, offset_map)
    boundaries = build_page_boundaries(offset_map)
    page_annotations = assign_annotations_to_pages(annotations, boundaries)

    nlp = spacy.blank("nl")
    db = DocBin().from_disk(str(spacy_file))
    pred_docs = list(db.get_docs(nlp.vocab))

    # Match pred docs to pages (stored in sorted page order)
    sorted_boundaries = sorted(boundaries, key=lambda x: x[0])
    if len(pred_docs) != len(sorted_boundaries):
        print(
            f"Warning: {len(pred_docs)} docs but {len(sorted_boundaries)} pages in offset map"
        )

    # 3. Build examples and collect spans for relaxed scoring
    examples = []
    all_gold_spans: list[tuple[int, int, str]] = []
    all_pred_spans: list[tuple[int, int, str]] = []
    pages_gold: list[list[tuple[int, int, str, str]]] = []
    pages_pred: list[list[tuple[int, int, str, str]]] = []
    page_numbers: list[int] = []

    for doc_idx in range(min(len(pred_docs), len(sorted_boundaries))):
        pred_doc = pred_docs[doc_idx]
        page, _, _ = sorted_boundaries[doc_idx]
        gold_spans = page_annotations.get(page, [])

        gold_doc = nlp(pred_doc.text)
        gold_ents = []
        for span_info in gold_spans:
            span = gold_doc.char_span(
                span_info["start"],
                span_info["end"],
                label=span_info["label"],
                alignment_mode="expand",
            )
            if span is not None:
                gold_ents.append(span)
                all_gold_spans.append((span.start_char, span.end_char, span.label_))

        pages_gold.append([(s.start_char, s.end_char, s.label_, s.text) for s in gold_ents])

        # spaCy disallows overlapping entities; filter to non-overlapping subset.
        # all_gold_spans retains the full set for relaxed scoring.
        non_overlapping = []
        for ent in gold_ents:
            if not any(
                _spans_overlap(ent.start_char, ent.end_char, e.start_char, e.end_char)
                for e in non_overlapping
            ):
                non_overlapping.append(ent)
        if len(non_overlapping) < len(gold_ents):
            print(
                f"  Page {page}: dropped {len(gold_ents) - len(non_overlapping)}"
                f" overlapping gold entities (kept {len(non_overlapping)})"
            )
        gold_doc.ents = non_overlapping

        for ent in pred_doc.ents:
            all_pred_spans.append((ent.start_char, ent.end_char, ent.label_))

        pages_pred.append([(e.start_char, e.end_char, e.label_, e.text) for e in pred_doc.ents])
        page_numbers.append(page)

        examples.append(Example(pred_doc, gold_doc))

    all_labels = {l for _, _, l in all_gold_spans} | {l for _, _, l in all_pred_spans}

    # 4. Strict scoring (exact span match)
    scorer = Scorer()
    strict = scorer.score(examples)

    # 5. Relaxed scoring (span overlap = match)
    relaxed = compute_relaxed_scores(all_gold_spans, all_pred_spans)
    relaxed_per_label = compute_relaxed_per_label(all_gold_spans, all_pred_spans, all_labels)

    # 6. Report
    print("\n" + "=" * 60)
    print("NER Evaluation Results")
    print("=" * 60)
    print(f"Gold annotations:  {len(all_gold_spans)}")
    print(f"Predicted entities: {len(all_pred_spans)}")
    print()

    header = f"{'':<30} {'Strict':>10} {'Relaxed':>10}"
    print(header)
    print("-" * len(header))

    for label in sorted(all_labels):
        s = strict.get("ents_per_type", {}).get(label, {})
        r = relaxed_per_label.get(label, {})
        print(f"{label}:")
        print(f"  {'P':<28} {s.get('p', 0):>10.3f} {r.get('p', 0):>10.3f}")
        print(f"  {'R':<28} {s.get('r', 0):>10.3f} {r.get('r', 0):>10.3f}")
        print(f"  {'F1':<28} {s.get('f', 0):>10.3f} {r.get('f', 0):>10.3f}")

    print("-" * len(header))
    print(f"{'Overall':<30} {strict.get('ents_f', 0):>10.3f} {relaxed['f']:>10.3f}")

    # 7. Export TP/FP/FN instances to CSV
    ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    tp, fp, fn = classify_instances(pages_gold, pages_pred, page_numbers)
    csv_path = ERRORS_DIR / (spacy_file.stem + "_errors.csv")
    write_errors_csv(tp, fp, fn, csv_path)

    # 8. Load meta sidecar for duration
    meta_path = spacy_file.with_name(spacy_file.stem + "_meta.json")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        duration_seconds = meta.get("duration_seconds", "")
        num_pages = meta.get("num_pages", "")
        inference_type = meta.get("ollama_host", "")
        failed_pages = meta.get("failed_pages", [])
    except (FileNotFoundError, json.JSONDecodeError):
        duration_seconds = ""
        num_pages = ""
        inference_type = ""
        failed_pages = []
    count_failed_pages = len(failed_pages)

    # 9. Export scores CSV — cumulative, appended each run
    source_text, model_name, prompting_method, temperature, prompt_language = _parse_spacy_filename(spacy_file)
    scores_path = EVAL_DIR / "scores.csv"
    _append_scores_csv(scores_path, source_text, model_name, prompting_method, temperature,
                       duration_seconds, num_pages, inference_type,
                       count_failed_pages,
                       all_labels, strict, relaxed_per_label, relaxed,
                       len(tp), len(fp), len(fn),
                       prompt_language=prompt_language)


def _parse_spacy_filename(spacy_file: Path):
    """Parse a .spacy filename into (source_text, model_name, prompting_method, temperature).

    Expects the naming convention from ner.py:
        {source_text}__{model}_t{TEMP}_{mode}.spacy
    e.g. 1816_third_letter__gemma4:31b_t0.1_fewshot.spacy

    The double underscore (__) between source text and model is the key delimiter,
    since source text itself may contain single underscores.
    """
    stem = spacy_file.stem  # removes .spacy

    try:
        # Split on __ to isolate source_text, then parse the rest
        source_text, rest = stem.split("__", maxsplit=1)

        # rest: {model}_t{TEMP}_{mode}_{language} — last 3 underscore segments are temp, mode, language
        *model_parts, temp_str, prompting_method, prompt_language = rest.rsplit("_", 3)
        temperature = temp_str[1:]  # strip the 't' prefix
        model_name = "_".join(model_parts)  # reassemble in case model has underscores

        source_text = f"{source_text}.txt"
    except (ValueError, IndexError):
        return (spacy_file.name, spacy_file.name, "unknown", "unknown", "unknown")

    return (source_text, model_name, prompting_method, temperature, prompt_language)


def _append_scores_csv(
    path: Path, source_text: str, model_name: str, prompting_method: str,
    temperature: str, duration_seconds: str | float,
    pages_processed: str | int, inference_type: str,
    count_failed_pages: int,
    all_labels: set,
    strict: dict, relaxed_per_label: dict, relaxed_overall: dict,
    count_tp: int = 0, count_fp: int = 0, count_fn: int = 0,
    prompt_language: str = "",
) -> None:
    """Append a summary row per label (+ overall) to a cumulative scores CSV.

    Creates the file with a header row if it doesn't exist yet; otherwise
    appends data rows only, so multiple evaluation runs accumulate into one file.
    Columns: source_text, model, temperature, prompting_method, prompt_language,
             pages_processed, duration_seconds, inference_type, datetime, label,
             strict_p, strict_r, strict_f1, relaxed_p, relaxed_r, relaxed_f1.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fieldnames = [
        "source_text", "model", "temperature", "prompting_method",
        "prompt_language",
        "pages_processed", "duration_seconds", "inference_type",
        "count_failed_pages",
        "datetime", "label",
        "strict_p", "strict_r", "strict_f1",
        "relaxed_p", "relaxed_r", "relaxed_f1",
        "count_tp", "count_fp", "count_fn",
    ]
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for label in sorted(all_labels):
            s = strict.get("ents_per_type", {}).get(label, {})
            r = relaxed_per_label.get(label, {})
            writer.writerow({
                "source_text": source_text,
                "model": model_name,
                "temperature": temperature,
                "prompting_method": prompting_method,
                "prompt_language": prompt_language,
                "duration_seconds": duration_seconds,
                "pages_processed": pages_processed,
                "inference_type": inference_type,
                "count_failed_pages": count_failed_pages,
                "datetime": now,
                "label": label,
                "strict_p": f"{s.get('p', 0):.4f}",
                "strict_r": f"{s.get('r', 0):.4f}",
                "strict_f1": f"{s.get('f', 0):.4f}",
                "relaxed_p": f"{r.get('p', 0):.4f}",
                "relaxed_r": f"{r.get('r', 0):.4f}",
                "relaxed_f1": f"{r.get('f', 0):.4f}",
            })

        # Overall row
        writer.writerow({
            "source_text": source_text,
            "model": model_name,
            "temperature": temperature,
            "prompting_method": prompting_method,
            "prompt_language": prompt_language,
            "duration_seconds": duration_seconds,
            "pages_processed": pages_processed,
            "inference_type": inference_type,
            "count_failed_pages": count_failed_pages,
            "datetime": now,
            "label": "OVERALL",
            "strict_p": f"{strict.get('ents_p', 0):.4f}",
            "strict_r": f"{strict.get('ents_r', 0):.4f}",
            "strict_f1": f"{strict.get('ents_f', 0):.4f}",
            "relaxed_p": f"{relaxed_overall.get('p', 0):.4f}",
            "relaxed_r": f"{relaxed_overall.get('r', 0):.4f}",
            "relaxed_f1": f"{relaxed_overall.get('f', 0):.4f}",
            "count_tp": count_tp,
            "count_fp": count_fp,
            "count_fn": count_fn,
        })

    print(f"Scores CSV written to {path}")


if __name__ == "__main__":
    main()

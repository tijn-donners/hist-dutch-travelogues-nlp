"""Evaluate NER predictions against Recogito manual annotations.

Loads a predicted .spacy DocBin and a Recogito JSON-LD export, aligns them
via full-text character offsets, and reports precision/recall/F1 per label.
"""

import csv
import json
import sys
from pathlib import Path
from spacy.scorer import Scorer
from spacy.training import Example
from spacy.tokens import DocBin
import spacy

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "ner-results"
RECOGITO_FILE = str(SCRIPT_DIR / "gs-annotations.jsonld")

TAG_MAP = {
    "E53 Place": "E53_Place",
    "E19 Physical Thing": "E19_Physical_Thing",
}


def select_results_file() -> Path:
    spacy_files = sorted(RESULTS_DIR.glob("*.spacy"))
    if not spacy_files:
        print("No .spacy files found in ner-results/")
        sys.exit(1)
    print("Available .spacy files:")
    for i, f in enumerate(spacy_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {i:>3}. {f.name}  ({size_mb:.1f} MB)")
    try:
        choice = input(f"\nSelect a file (1-{len(spacy_files)}): ").strip()
        idx = int(choice) - 1
        if idx < 0 or idx >= len(spacy_files):
            raise ValueError
    except (ValueError, EOFError, KeyboardInterrupt):
        print("Invalid selection.")
        sys.exit(1)
    return spacy_files[idx]


def load_recogito_annotations(path: str) -> list[dict]:
    """Extract annotations with purpose=tagging from Recogito JSON-LD."""
    with open(path) as f:
        data = json.load(f)

    annotations = []
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

        tag = TAG_MAP.get(tag, tag)
        selector = item.get("target", {}).get("selector", [])
        pos = None
        for sel in selector:
            if sel.get("type") == "TextPositionSelector":
                pos = (sel["start"], sel["end"])
                break
        if pos is None:
            continue

        annotations.append({"start": pos[0], "end": pos[1], "label": tag})

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
    """Per-page relaxed matching: classify each span as TP, FP, or FN."""
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
    """Write classified instances to a CSV, sorted by page then type."""
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


def main():
    # 1. Select files
    spacy_file = select_results_file()

    offset_map_path = spacy_file.with_name(
        spacy_file.stem.replace("all_pages", "offset_map") + ".json"
    )
    if not offset_map_path.exists():
        print(f"No offset map found: {offset_map_path}")
        print("Re-run ner.py to generate one (markers are now kept in chunk text).")
        sys.exit(1)

    if not Path(RECOGITO_FILE).exists():
        print(f"Recogito annotations file not found: {RECOGITO_FILE}")
        sys.exit(1)

    # 2. Load data
    with open(offset_map_path) as f:
        offset_map = json.load(f)

    annotations = load_recogito_annotations(RECOGITO_FILE)
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

    # 5. Relaxed scoring (overlap = match)
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
    tp, fp, fn = classify_instances(pages_gold, pages_pred, page_numbers)
    csv_path = spacy_file.with_name(spacy_file.stem + "_errors.csv")
    write_errors_csv(tp, fp, fn, csv_path)


if __name__ == "__main__":
    main()

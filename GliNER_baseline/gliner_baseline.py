"""GLiNER v1 multilingual baseline NER for 19th-century Dutch travelogues.

Uses ``urchade/gliner_multi-v2.1`` (XLM-RoBERTa, multilingual incl. Dutch) — a
Dutch-capable encoder, unlike the prior English-only GLiNER2 baseline. GLiNER v1
takes a flat list of label strings; we feed readable aliases and map predictions
back to the 8 canonical CIDOC labels (see ``labels.py``).

Produces output matching the ``ner/ner.py`` I/O contract so the existing
``ner/ner_evaluate.py`` scores it with zero changes:

  ner/ner-output/gliner_multi/{short_label}/
      {letter_label}__gliner_multi_t0.0_zeroshot_{language}.spacy
      {letter_label}__gliner_multi_t0.0_zeroshot_{language}_offset_map.json
      {letter_label}__gliner_multi_t0.0_zeroshot_{language}_meta.json

Page splitting and offset computation are reused from ``ner/ner.py`` so char
offsets match the LLM runs byte-for-byte (the evaluator's gold alignment relies
on this).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
NER_DIR = ROOT_DIR / "ner"
sys.path.insert(0, str(NER_DIR))  # so we can import helpers from ner/ner.py

# Reuse the existing page-loading + offset + label logic (do not reimplement).
from ner import (  # noqa: E402
    load_pages_from_txt,
    load_pages_from_txt_horizontal_rule,
    _short_label_from_path,
    _letter_label_from_path,
    _page_sort_key,
    OUTPUT_DIR as NER_OUTPUT_DIR,
    INPUT_MODE,
)
import spacy  # noqa: E402
from spacy.tokens import DocBin  # noqa: E402

from labels import ALIAS_TO_CIDOC  # noqa: E402

MODEL_NAME = "gliner_multi"  # short tag used in the output filename (eval parses this)
DEFAULT_GLINER_MODEL = "urchade/gliner_multi-v2.1"
DEFAULT_THRESHOLD = 0.5
TEMPERATURE_TAG = "0.0"  # GLiNER has no temperature; 0.0 for filename-convention compliance
MODE = "zeroshot"        # GLiNER is zero-shot by nature


def parse_args():
    p = argparse.ArgumentParser(
        description="Run GLiNER v1 multilingual zero-shot NER on travelogue text, "
                    "emitting the ner.py sidecar contract."
    )
    p.add_argument("--input", "-i", default="data/1816_third_letter.txt",
                   help="Input .txt (relative to repo root). Default: data/1816_third_letter.txt")
    p.add_argument("--split-mode", choices=["brackets", "horizontal-rule"], default="brackets",
                   help="Page-split strategy (must match ner.py). Default: brackets ([N] markers)")
    p.add_argument("--language", default="dutch",
                   help="Prompt language tag for filename contract. Default: dutch")
    p.add_argument("--gliner-model", default=DEFAULT_GLINER_MODEL,
                   help=f"HuggingFace GLiNER checkpoint. Default: {DEFAULT_GLINER_MODEL}")
    p.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                   help="Confidence threshold for keeping entities. Default: 0.5")
    p.add_argument("--max-len", type=int, default=512,
                   help="Max token length per page (GLiNER default 384 truncates long "
                        "pages; 512 = XLM-R limit, covers the corpus). Default: 512")
    p.add_argument("--no-eval", action="store_true",
                   help="Do not run ner_evaluate.py afterwards (run_baseline.sh only)")
    return p.parse_args()


def _spans_overlap(a_start, a_end, b_start, b_end):
    return not (a_end <= b_start or b_end <= a_start)


def _greedy_non_overlapping(spans):
    """Filter to a non-overlapping subset, keeping longer spans first.

    GLiNER v1 with ``flat_ner=True`` already removes overlaps, but spaCy
    disallows overlapping entities in ``doc.ents`` so we keep a safety filter
    (mirrors ner_evaluate.py's overlap-filter intent).
    """
    ordered = sorted(spans, key=lambda s: (s.end_char - s.start_char), reverse=True)
    kept = []
    for s in ordered:
        if not any(_spans_overlap(s.start_char, s.end_char, k.start_char, k.end_char)
                   for k in kept):
            kept.append(s)
    return kept


def main():
    args = parse_args()
    os.chdir(SCRIPT_DIR)  # mirrors ner.py: relative paths resolve against script dir

    txt_file = str(ROOT_DIR / args.input)
    if not Path(txt_file).exists():
        print(f"ERROR: input file not found: {txt_file}")
        sys.exit(1)

    if INPUT_MODE == "pagexml":
        print("ERROR: PageXML input mode is not supported by the GliNER baseline. "
              "Use a .txt source (--input data/...txt).")
        sys.exit(1)

    print(f"Loading text from: {txt_file}")
    if args.split_mode == "horizontal-rule":
        print("  Split mode: horizontal rules ('____________')")
        pages, _ = load_pages_from_txt_horizontal_rule(txt_file)
    else:
        print("  Split mode: bracket markers ([N])")
        pages, _ = load_pages_from_txt(txt_file)

    print(f"Loaded {len(pages)} pages")
    if not pages:
        print("ERROR: No pages were loaded from the input file.")
        sys.exit(1)

    # Limit torch threads to leave headroom on the 4-core box (RAM is the binding constraint).
    try:
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    except Exception:
        pass

    print(f"Loading GLiNER model: {args.gliner_model} (CPU) ...")
    t_load = time.perf_counter()
    from gliner import GLiNER
    model = GLiNER.from_pretrained(args.gliner_model)
    # GLiNER's default max_len=384 truncates long pages (corpus pages run ~400-435 tokens).
    # The processor reads config.max_len live, so raise it to the XLM-R limit (512).
    model.config.max_len = args.max_len
    print(f"Model loaded in {round(time.perf_counter() - t_load, 1)}s (max_len={model.config.max_len})")

    nlp = spacy.blank("nl")  # matches the evaluator's vocab
    aliases = list(ALIAS_TO_CIDOC.keys())
    all_docs = []
    offset_map = {}
    failed_pages = []
    sorted_pages = sorted(pages.items(), key=lambda x: _page_sort_key(x[0]))
    total_pages = len(sorted_pages)

    print(f"Starting zero-shot NER (threshold={args.threshold}, "
          f"labels={len(aliases)} aliases -> {len(set(ALIAS_TO_CIDOC.values()))} CIDOC)\n")
    t_start = time.perf_counter()
    for i, (pagenumber, (page_offset, page_text)) in enumerate(sorted_pages, 1):
        print(f"[{i}/{total_pages}] Processing page {pagenumber}...", end=" ", flush=True)
        try:
            ents = model.predict_entities(
                page_text, aliases, flat_ner=True, threshold=args.threshold,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            doc = nlp.make_doc(page_text)
            failed_pages.append(pagenumber)
            all_docs.append(doc)
            offset_map[pagenumber] = page_offset
            continue

        doc = nlp.make_doc(page_text)
        spans = []
        for e in ents:
            canonical = ALIAS_TO_CIDOC.get(e.get("label", ""))
            if canonical is None:
                continue  # unknown label — skip
            start = e.get("start"); end = e.get("end")
            if start is None or end is None or start < 0 or end > len(page_text) or start >= end:
                continue
            span = doc.char_span(int(start), int(end), label=canonical, alignment_mode="expand")
            if span is not None:
                spans.append(span)
        spans = _greedy_non_overlapping(spans)
        try:
            doc.ents = spans
        except ValueError:
            doc.ents = _greedy_non_overlapping(spans)
        print(f"done ({len(doc.ents)} entities)", flush=True)

        all_docs.append(doc)
        offset_map[pagenumber] = page_offset
    duration_seconds = round(time.perf_counter() - t_start, 1)

    # ---- Write sidecars (filename convention consumed by ner_evaluate.py) ----
    letter_label = _letter_label_from_path(args.input)
    short_label = _short_label_from_path(args.input)
    base = f"{letter_label}__{MODEL_NAME}_t{TEMPERATURE_TAG}_{MODE}_{args.language}"

    out_subdir = NER_OUTPUT_DIR / MODEL_NAME / short_label
    out_subdir.mkdir(parents=True, exist_ok=True)

    meta_path = out_subdir / f"{base}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "model": args.gliner_model,
            "mode": MODE,
            "language": args.language,
            "temperature": float(TEMPERATURE_TAG),
            "threshold": args.threshold,
            "source_text": args.input,
            "duration_seconds": duration_seconds,
            "num_pages": total_pages,
            "ollama_host": "cpu",  # surfaced as inference_type in scores.csv
            "failed_pages": failed_pages,
        }, f, indent=2)
    print(f"\nMetadata saved to: {meta_path}")

    spacy_path = out_subdir / f"{base}.spacy"
    DocBin(docs=all_docs, store_user_data=True).to_disk(spacy_path)
    print(f"Merged DocBin saved to: {spacy_path}")

    offset_map_path = out_subdir / f"{base}_offset_map.json"
    with open(offset_map_path, "w") as f:
        json.dump(offset_map, f)
    print(f"Offset map saved to: {offset_map_path}")

    print(f"\nDone in {duration_seconds}s — {sum(len(d.ents) for d in all_docs)} entities, "
          f"{len(failed_pages)} failed pages.")
    print(f"\nEvaluate with:\n  python {NER_DIR / 'ner_evaluate.py'} -s {spacy_path}")


if __name__ == "__main__":
    main()
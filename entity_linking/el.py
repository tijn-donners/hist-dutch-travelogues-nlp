"""Entity Linking pipeline for 19th-century Dutch travelogue NER results.

Three-stage LELA-inspired architecture:
  1. Candidate Generation (Wikidata + GeoNames APIs)
  2. Candidate Reranking (cross-encoder or heuristic)
  3. LLM Selection (Ollama with domain-aware Dutch prompt)

Loads .spacy files from NER, runs EL, and saves results with kb_ids.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import spacy
from spacy.tokens import Span, Token
from dotenv import load_dotenv
from spacy.tokens import DocBin

# Register custom extension attributes for entities and tokens
if not Span.has_extension("kb_id_wikidata_"):
    Span.set_extension("kb_id_wikidata_", default=None)
if not Token.has_extension("ent_kb_id_wikidata_"):
    Token.set_extension("ent_kb_id_wikidata_", default=None)
if not Span.has_extension("kb_id_geonames_"):
    Span.set_extension("kb_id_geonames_", default=None)
if not Token.has_extension("ent_kb_id_geonames_"):
    Token.set_extension("ent_kb_id_geonames_", default=None)

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

from el_candidates import generate_candidates, clear_cache
from el_reranker import rerank_candidates
from el_selector import select_candidate

# ── Configuration ──────────────────────────────────────────────────────────

NER_RESULTS_DIR = ROOT_DIR / "ner" / "ner-output"
EL_OUTPUT_DIR = ROOT_DIR / "entity_linking" / "el-results"
MODEL_NAME = "gemma4:31b"
TOP_K_RERANK = 3

if os.environ.get("OLLAMA_API_KEY"):
    OLLAMA_URL = "https://ollama.com"
    OLLAMA_HEADERS = {"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"}
else:
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_HEADERS = None


# ── File selection ─────────────────────────────────────────────────────────

def select_input_file():
    """Scan ner-output/ for .spacy files and let the user pick one.

    Returns:
        Path to the selected .spacy file.
    """
    spacy_files = sorted(
        p for p in NER_RESULTS_DIR.rglob("*.spacy") if "_el" not in p.stem
    )
    if not spacy_files:
        print(f"No .spacy files found in {NER_RESULTS_DIR}")
        raise SystemExit(1)

    if len(spacy_files) == 1:
        print(f"Auto-selected: {spacy_files[0].relative_to(NER_RESULTS_DIR)}")
        return str(spacy_files[0])

    print("Available .spacy files:")
    for i, f in enumerate(spacy_files, 1):
        rel = f.relative_to(NER_RESULTS_DIR)
        print(f"  [{i}] {rel}")
    choice = input("Select number: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(spacy_files):
            return str(spacy_files[idx])
    except ValueError:
        pass
    print(f"Invalid selection: {choice}")
    raise SystemExit(1)
def extract_context(doc_text: str, start_char: int, end_char: int,
                    window: int | None = None) -> str:
    """Extract context around a character-offset entity span.

    Args:
        doc_text: Full page text.
        start_char: Start character offset of the entity.
        end_char: End character offset of the entity.
        window: Characters on each side (default None = entire page).

    Returns:
        Context string with newlines replaced by spaces.
    """
    if window is None:
        return doc_text.replace('\n', ' ')
    ctx_start = max(0, start_char - window)
    ctx_end = min(len(doc_text), end_char + window)
    context = doc_text[ctx_start:ctx_end].replace('\n', ' ')
    return context


def _parse_think_arg(value: str | None) -> bool | str | None:
    """Parse the --think CLI argument into a valid think parameter value.

    Args:
        value: Raw string from argparse (None if not provided).

    Returns:
        None for model default, False for "false", or "low"/"medium"/"high".
    """
    if value is None:
        return None
    value_lower = value.strip().lower()
    if value_lower == "false":
        return False
    if value_lower == "true":
        return True
    if value_lower in ("low", "medium", "high"):
        return value_lower
    print(f"Warning: unrecognized --think value '{value}', using model default.")
    return None


def link_entities(
    spacy_file: str,
    model_name: str = "gemma4:31b-cloud",
    top_k_rerank: int = 3,
    ollama_url: str = "http://localhost:11434",
    ollama_headers: dict | None = None,
    think: bool | str | None = None,
    temperature: float = 0.0,
) -> tuple[list, dict]:
    """Run the full EL pipeline on a .spacy file.

    Args:
        spacy_file: Path to the .spacy DocBin file with NER annotations.
        model_name: Ollama model name for Stage 2 reranking and Stage 3 selection.
        top_k_rerank: Number of candidates to retain after reranking.
        ollama_url: Ollama server base URL.
        ollama_headers: Optional auth headers for cloud API (Bearer token).
        think: Thinking mode (True, False, "low", "medium", "high").
               None uses the model's default. Set to False or "low" for
               thinking models that over-think on simple scoring tasks.

    Returns:
        Tuple of (list of updated spaCy Docs with kb_id_ attributes set,
                   dict of pipeline statistics).
    """
    nlp = spacy.blank("nl")
    db = DocBin().from_disk(spacy_file)
    docs = list(db.get_docs(nlp.vocab))

    # Pre-count total entities for progress reporting
    total_ents = sum(len(doc.ents) for doc in docs)
    linked = 0
    processed = 0
    count_skipped = 0
    count_errors = 0

    for doc in docs:
        for ent in doc.ents:
            processed += 1
            print(f"\n--- [{processed}/{total_ents}] {ent.text} ({ent.label_}) ---")
            context = extract_context(doc.text, ent.start_char, ent.end_char)

            # Skip EL for generic-type labels — no KB linking
            if ent.label_ in ("Mode_of_Transportation", "E52_Time_Span", "F2_Expression",
                              "E19_Physical_Object", "E20_Biological_Object", "E31_Document"):
                ent._.kb_id_wikidata_ = None
                ent._.kb_id_geonames_ = None
                for token in ent:
                    token._.ent_kb_id_wikidata_ = None
                    token._.ent_kb_id_geonames_ = None
                if ent.label_ == "E52_Time_Span":
                    label_name = "time span"
                elif ent.label_ == "F2_Expression":
                    label_name = "artistic expression"
                elif ent.label_ == "E19_Physical_Object":
                    label_name = "physical object"
                elif ent.label_ == "E20_Biological_Object":
                    label_name = "biological specimen"
                elif ent.label_ == "E31_Document":
                    label_name = "document"
                else:
                    label_name = "generic transport type"
                print(f"  -> {ent.text} ({ent.label_}): skipped EL ({label_name})")
                count_skipped += 1
                continue

            # Stage 1: Candidate Generation
            candidates_by_kb = generate_candidates(
                ent.text,
                ent.label_,
                context=context,
                ollama_url=ollama_url,
                model_name=model_name,
                ollama_headers=ollama_headers,
                think=think,
                temperature=temperature,
            )

            wikidata_candidates = candidates_by_kb.get("wikidata", [])
            geonames_candidates = candidates_by_kb.get("geonames", [])

            if not wikidata_candidates and not geonames_candidates:
                continue

            # Stage 2: Process Wikidata candidates
            wikidata_selected_id = "NIL"
            wikidata_reranked = []
            if wikidata_candidates:
                # Rerank candidates
                try:
                    wikidata_reranked = rerank_candidates(
                        ent.text,
                        context,
                        wikidata_candidates,
                        top_k=top_k_rerank,
                        ollama_url=ollama_url,
                        model_name=model_name,
                        ollama_headers=ollama_headers,
                        think=think,
                        temperature=temperature,
                    )
                    # Select best Wikidata candidate
                    wikidata_selected_id = select_candidate(
                        ent.text,
                        context,
                        wikidata_reranked,
                        model_name=model_name,
                        ollama_url=ollama_url,
                        ollama_headers=ollama_headers,
                        think=think,
                        temperature=temperature,
                    )
                except Exception as e:
                    err = str(e)
                    if "timeout" in err.lower() or "timed out" in err.lower():
                        wikidata_selected_id = "ERROR:Ollama Wikidata timed out"
                    else:
                        wikidata_selected_id = f"ERROR:Ollama Wikidata failed ({err})"
                    print(f"  -> Wikidata: {wikidata_selected_id}")

            # Stage 2: Process GeoNames candidates
            geonames_selected_id = "NIL"
            geonames_reranked = []
            if geonames_candidates:
                # Rerank candidates
                try:
                    geonames_reranked = rerank_candidates(
                        ent.text,
                        context,
                        geonames_candidates,
                        top_k=top_k_rerank,
                        ollama_url=ollama_url,
                        model_name=model_name,
                        ollama_headers=ollama_headers,
                        think=think,
                        temperature=temperature,
                    )
                    # Select best GeoNames candidate
                    geonames_selected_id = select_candidate(
                        ent.text,
                        context,
                        geonames_reranked,
                        model_name=model_name,
                        ollama_url=ollama_url,
                        ollama_headers=ollama_headers,
                        think=think,
                        temperature=temperature,
                    )
                except Exception as e:
                    err = str(e)
                    if "timeout" in err.lower() or "timed out" in err.lower():
                        geonames_selected_id = "ERROR:Ollama GeoNames timed out"
                    else:
                        geonames_selected_id = f"ERROR:Ollama GeoNames failed ({err})"
                    print(f"  -> GeoNames: {geonames_selected_id}")

            # Stage 3: Set KB IDs on entity and tokens
            linked_this_entity = False
            if wikidata_selected_id.startswith("ERROR:"):
                ent._.kb_id_wikidata_ = wikidata_selected_id
                for token in ent:
                    token._.ent_kb_id_wikidata_ = wikidata_selected_id
                print(f"  -> Wikidata: {wikidata_selected_id}")
                count_errors += 1
            elif wikidata_selected_id != "NIL":
                ent._.kb_id_wikidata_ = wikidata_selected_id
                for token in ent:
                    token._.ent_kb_id_wikidata_ = wikidata_selected_id
                linked_this_entity = True
                selected_label = next(
                    (c["label"] for c in wikidata_reranked if c["id"] == wikidata_selected_id),
                    "",
                ) if wikidata_candidates else ""
                print(f"  -> Wikidata: {wikidata_selected_id} ({selected_label})")
            else:
                ent._.kb_id_wikidata_ = None
                for token in ent:
                    token._.ent_kb_id_wikidata_ = None

            if geonames_selected_id.startswith("ERROR:"):
                ent._.kb_id_geonames_ = geonames_selected_id
                for token in ent:
                    token._.ent_kb_id_geonames_ = geonames_selected_id
                print(f"  -> GeoNames: {geonames_selected_id}")
                count_errors += 1
            elif geonames_selected_id != "NIL":
                ent._.kb_id_geonames_ = geonames_selected_id
                for token in ent:
                    token._.ent_kb_id_geonames_ = geonames_selected_id
                linked_this_entity = True
                selected_label = next(
                    (c["label"] for c in geonames_reranked if c["id"] == geonames_selected_id),
                    "",
                ) if geonames_candidates else ""
                print(f"  -> GeoNames: {geonames_selected_id} ({selected_label})")
            else:
                ent._.kb_id_geonames_ = None
                for token in ent:
                    token._.ent_kb_id_geonames_ = None

            if linked_this_entity:
                linked += 1

            if not (wikidata_selected_id != "NIL" or geonames_selected_id != "NIL"):
                print(f"  -> NIL (no match in either KB)")

    # Build pipeline statistics
    count_linked_wd = 0
    count_linked_gn = 0
    count_linked_both = 0
    count_linked_neither = 0
    for doc in docs:
        for ent in doc.ents:
            wd = ent._.kb_id_wikidata_
            gn = ent._.kb_id_geonames_
            is_err_wd = isinstance(wd, str) and wd.startswith("ERROR:")
            is_err_gn = isinstance(gn, str) and gn.startswith("ERROR:")
            has_wd = wd is not None and not is_err_wd
            has_gn = gn is not None and not is_err_gn
            if has_wd:
                count_linked_wd += 1
            if has_gn:
                count_linked_gn += 1
            if has_wd and has_gn:
                count_linked_both += 1
            if not has_wd and not has_gn:
                count_linked_neither += 1

    stats = {
        "total_entities": total_ents,
        "count_errors": count_errors,
        "count_skipped": count_skipped,
        "count_linked_wd": count_linked_wd,
        "count_linked_gn": count_linked_gn,
        "count_linked_both": count_linked_both,
        "count_linked_neither": count_linked_neither,
    }

    print(f"\nLinked {linked}/{total_ents} entities to at least one KB ({linked/total_ents*100:.1f}%)"
          if total_ents else "\nNo entities found.")
    return docs, stats


# ── Run ────────────────────────────────────────────────────────────────────

def main():
    """Run the full Entity Linking pipeline and save results.

    Interactively selects a .spacy file from ner-output/, runs the three-stage
    pipeline, and writes the enriched DocBin to entity_linking/el-results/.
    Copies the matching offset map alongside the output.
    """
    parser = argparse.ArgumentParser(
        description="Run Entity Linking pipeline on a .spacy file."
    )
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"LLM model name (default: {MODEL_NAME})")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM temperature (default: 0.0)")
    parser.add_argument("--top-k", type=int, default=TOP_K_RERANK,
                        help=f"Top-k rerank candidates (default: {TOP_K_RERANK})")
    parser.add_argument("--think", type=str, default=None,
                        help="Thinking mode: false, low, medium, high "
                             "(default: model's default). Use false/low for "
                             "thinking models that over-think on simple tasks.")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input .spacy file (skip interactive selection)")
    args = parser.parse_args()

    model_name = args.model
    temperature = args.temperature
    top_k = args.top_k
    think = _parse_think_arg(args.think)

    if args.input:
        spacy_file = Path(args.input)
        if not spacy_file.exists():
            print(f"Input file not found: {spacy_file}")
            sys.exit(1)
    else:
        spacy_file = select_input_file()
    print(f"Loading: {spacy_file}")
    print(f"LLM: {model_name}")
    print(f"Temperature: {temperature}")
    print(f"Ollama: {OLLAMA_URL} {'(cloud)' if OLLAMA_HEADERS else '(local)'}")
    print(f"GeoNames: enabled (gazetteer)")
    print(f"Top-k rerank: {top_k}")
    if think is not None:
        print(f"Think mode: {think}")
    print()

    clear_cache()

    t0 = time.time()
    docs, stats = link_entities(
        spacy_file=spacy_file,
        model_name=model_name,
        top_k_rerank=top_k,
        ollama_url=OLLAMA_URL,
        ollama_headers=OLLAMA_HEADERS,
        think=think,
        temperature=temperature,
    )
    t1 = time.time()
    duration = t1 - t0

    # Write EL output to el-results/ with EL config in filename
    EL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(spacy_file).stem
    model_slug = model_name.replace(":", "-").replace("/", "-")
    think_slug = f"_think{str(think).lower()}" if think is not None else ""
    output_path = EL_OUTPUT_DIR / f"{stem}__{model_slug}_t{temperature}{think_slug}_el.spacy"
    merged_docbin = DocBin(docs=docs, store_user_data=True)
    merged_docbin.to_disk(str(output_path))
    print(f"\nSaved to: {output_path}")

    # Copy matching offset map with the same stem
    spacy_path = Path(spacy_file)
    offset_stem = stem + "_offset_map"
    offset_map = spacy_path.parent / f"{offset_stem}.json"
    if offset_map.exists():
        dest = EL_OUTPUT_DIR / f"{stem}__{model_slug}_t{temperature}{think_slug}_offset_map.json"
        shutil.copy2(offset_map, dest)
        print(f"Offset map copied to: {dest}")
    else:
        print(f"Warning: no offset map found ({offset_map.name})")

    # Save run info for evaluation to pick up later
    info = {
        "stats": stats,
        "duration": duration,
        "model": model_name,
        "temperature": temperature,
        "top_k": top_k,
        "think": str(think).lower() if think is not None else "",
        "inference_type": "cloud" if OLLAMA_HEADERS else "local",
    }
    info_path = EL_OUTPUT_DIR / f"{stem}__{model_slug}_t{temperature}{think_slug}_el_run_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f)
    print(f"Run info saved to: {info_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import re as regex
from dotenv import load_dotenv
from spacy_llm.util import assemble_from_config
from spacy.tokens import DocBin
from spacy.util import load_config

from streaming_patch import patch_langchain_ollama_streaming

# ---------------------------------------------------------------------------
# Page number sorting helper
# ---------------------------------------------------------------------------
_ROMAN_VALUES = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7,
    'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12, 'XIII': 13,
    'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18,
    'XIX': 19, 'XX': 20, 'XXI': 21, 'XXII': 22, 'XXIII': 23,
    'XXIV': 24, 'XXV': 25, 'XXVI': 26, 'XXVII': 27, 'XXVIII': 28,
    'XXIX': 29, 'XXX': 30, 'XXXI': 31, 'XXXII': 32, 'XXXIII': 33,
    'XXXIV': 34, 'XXXV': 35, 'XXXVI': 36, 'XXXVII': 37, 'XXXVIII': 38,
    'XXXIX': 39, 'XL': 40,
}

def _page_sort_key(pn):
    """Sort key: Roman numerals -> numeric, Arabic -> numeric, fallback -> 0."""
    s = str(pn)
    if s.isdigit():
        return int(s)
    if s.upper() in _ROMAN_VALUES:
        return _ROMAN_VALUES[s.upper()]
    return 0

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments for NER pipeline configuration."""
    parser = argparse.ArgumentParser(
        description="Run NER on 19th-century Dutch travelogue texts using spaCy-LLM + Ollama."
    )
    parser.add_argument(
        "--model", "-m",
        default="gemma4:31b",
        help="Ollama model name (default: gemma4:31b)"
    )
    parser.add_argument(
        "--mode", "-M",
        choices=["zeroshot", "fewshot"],
        default="fewshot",
        help="NER prompt strategy: fewshot or zeroshot (default: fewshot)"
    )
    parser.add_argument(
        "--ollama-host", "-H",
        choices=["cloud", "localhost"],
        default="cloud",
        help="Ollama API host: cloud (ollama.com) or localhost (default: cloud)"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/1816_third_letter.txt",
        help="Path to source .txt file, relative to project root (default: data/1816_third_letter.txt)"
    )
    parser.add_argument(
        "--split-mode", "-S",
        choices=["brackets", "horizontal-rule"],
        default="brackets",
        help="How to split the text into pages: 'brackets' uses [N] markers, "
             "'horizontal-rule' splits on lines of underscores (default: brackets)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        help="LLM temperature for NER inference as float (default: 0.0)"
    )
    parser.add_argument(
        "--language", "-L",
        choices=["dutch", "english"],
        default="dutch",
        help="Prompt language: dutch or english (default: dutch)"
    )
    return parser.parse_args()

# Fixed configuration (not exposed as CLI args)
INPUT_MODE = "txt"  # "txt" or "pagexml" (pagexml requires corrected files in page_updated/)
PAGEXML_DIR = str(ROOT_DIR / "data/page_updated/")
SCAN_CSV = str(ROOT_DIR / "data/1816-scannumber-to-pagenumber.csv")
OUTPUT_DIR = SCRIPT_DIR / "ner-output"

# Maps (language, mode) pair to config filename
_MODE_CONFIG = {
    "dutch": {
        "fewshot": "config/fewshot.cfg",
        "zeroshot": "config/zeroshot.cfg",
    },
    "english": {
        "fewshot": "config/fewshot_english.cfg",
        "zeroshot": "config/zeroshot_english.cfg",
    },
}


def _letter_label_from_path(txt_path):
    """Derive a label from the input filename.

    e.g. 'data/1816_third_letter.txt' -> '1816_third_letter'
    """
    return Path(txt_path).stem


def _short_label_from_path(txt_path):
    """Derive a short output subdirectory label from the input filename.

    e.g. 'data/1816_third_letter.txt' -> '1816'
    """
    return Path(txt_path).stem.split('_')[0]


# ---------------------------------------------------------------------------
# PageXML input helpers
# ---------------------------------------------------------------------------
def _ns(root):
    """Retrieve namespace from root element."""
    match = regex.match(r'\{([^}]+)\}', root.tag)
    return match.group(1) if match else ''


def get_pagexml_text_lines(xml_path):
    """Return list of text lines (in reading order) from a PageXML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = _ns(root)

    ordered_group = root.find(f".//{{{ns}}}OrderedGroup")
    if ordered_group is None:
        return []

    lines = []
    for ref in ordered_group.findall(f"{{{ns}}}RegionRefIndexed"):
        region_id = ref.attrib['regionRef']
        region = root.find(f".//{{{ns}}}TextRegion[@id='{region_id}']")
        if region is not None:
            for line_el in region.findall(f"{{{ns}}}TextLine"):
                text_el = line_el.find(f"{{{ns}}}TextEquiv/{{{ns}}}PlainText")
                text = text_el.text if text_el is not None and text_el.text else ""
                lines.append(text)
    return lines


def extract_scan_number(xml_filename):
    """Extract scan number from '0552_0179_0001.xml' -> 1."""
    stem = Path(xml_filename).stem
    parts = stem.split('_')
    return int(parts[-1])


def load_scan_page_mapping(csv_path):
    """Return dict: scan_number -> page_number (str because of roman numerals)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return dict(zip(df['Scan Number'], df['Page Number'].astype(str)))


# ---------------------------------------------------------------------------
# Load text depending on input mode
# ---------------------------------------------------------------------------
def load_pages_from_txt(txt_path):
    """Parse .txt with [N] markers, return (pages dict, None)."""
    with open(txt_path, 'r') as f:
        letter = f.read()

    marker_matches = list(regex.finditer(r'\[(\d+)\]', letter))
    pages = {}
    for i, m in enumerate(marker_matches):
        page_num = int(m.group(1))
        page_start = m.start()
        next_start = marker_matches[i + 1].start() if i + 1 < len(marker_matches) else len(letter)
        page_text = letter[page_start:next_start]
        if page_text.strip():
            pages[page_num] = (page_start, page_text)
    return pages, None


def load_pages_from_txt_horizontal_rule(txt_path):
    """Parse .txt by splitting on horizontal rules (lines of underscores).

    The '________________' rule is used in the printed book to separate
    recto/verso page pairs. This split mode assigns sequential integer
    page numbers (1, 2, 3, ...) to each chunk.

    Returns (pages dict, None).
    """
    with open(txt_path, 'r') as f:
        letter = f.read()

    # Strip BOM if present
    if letter.startswith('﻿'):
        letter = letter[1:]

    # Split on lines containing at least 16 underscores
    chunks = regex.split(r'\n_{16,}\n', letter)
    chunks = [c.strip() for c in chunks if c.strip()]

    pages = {}
    offset = 0
    for page_num, chunk in enumerate(chunks, start=1):
        pages[page_num] = (offset, chunk)
        offset += len(chunk)

    return pages, None


def load_pages_from_pagexml(pagexml_dir, scan_csv):
    """Parse corrected PageXML files, return (pages dict, line_offset_map).

    pages: {pagenumber: (full_text_offset, page_text)}
        page_text has lines joined with \\n.
        full_text_offset is a virtual offset (concatenation of all pages).

    line_offset_map: {pagenumber: {char_offset_within_page: line_index}}
        maps each character position within a page to the line it belongs to.
    """
    scan_to_page = load_scan_page_mapping(scan_csv)
    xml_files = sorted(Path(pagexml_dir).glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No .xml files found in {pagexml_dir}")

    # Build per-page data
    page_data = {}  # scan_number -> (page_number, line_texts)
    for xml_path in xml_files:
        scan_num = extract_scan_number(xml_path.name)
        page_num = scan_to_page.get(scan_num)
        if page_num is None:
            print(f"  Skipping {xml_path.name}: no scan→page mapping for scan {scan_num}")
            continue
        lines = get_pagexml_text_lines(str(xml_path))
        if not lines:
            print(f"  Skipping {xml_path.name}: no text lines found")
            continue
        page_data[scan_num] = (page_num, lines)
        print(f"  {xml_path.name}: scan {scan_num} → page {page_num} ({len(lines)} lines)")

    # Sort by scan number, build pages dict with virtual full-text offsets
    pages = {}
    line_offset_map = {}
    full_text_offset = 0
    for scan_num in sorted(page_data):
        page_num, lines = page_data[scan_num]
        page_text = "\n".join(lines)
        pages[int(page_num) if page_num.isdigit() else page_num] = (full_text_offset, page_text)

        # Build line offset map for this page
        char_pos = 0
        line_map = {}
        for li, line_text in enumerate(lines):
            for _ in line_text:
                line_map[char_pos] = li
                char_pos += 1
            # the newline belongs to the current line
            if li < len(lines) - 1:
                line_map[char_pos] = li
                char_pos += 1  # the \n character
        line_offset_map[page_num] = line_map

        full_text_offset += len(page_text)

    return pages, line_offset_map


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    """Run the NER pipeline on the travelogue text.

    Loads pages from .txt or corrected PageXML, assembles a spaCy-LLM pipeline
    from the configured NER config file, and runs NER per page. Saves merged DocBin,
    full-text offset map, and (in PageXML mode) a line offset map.
    """
    args = parse_args()
    load_dotenv()
    os.chdir(SCRIPT_DIR)  # config relative paths resolve against script dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    letter_label = _letter_label_from_path(args.input)
    short_label = _short_label_from_path(args.input)

    # Create model/letter subdirectory
    out_subdir = OUTPUT_DIR / args.model / short_label
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Select config file based on mode and language
    ner_config = _MODE_CONFIG[args.language][args.mode]
    config_path = str(SCRIPT_DIR / ner_config)
    config = load_config(config_path)

    # Override model name and API host in the loaded config
    config["components"]["llm"]["model"]["name"] = args.model
    config["components"]["llm"]["model"]["config"]["temperature"] = args.temperature
    # 10-minute timeout for the HTTP request (applies to connect + each read chunk)
    config["components"]["llm"]["model"]["config"]["timeout"] = 600

    if args.ollama_host == "cloud":
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            print("WARNING: --ollama-host=cloud but no OLLAMA_API_KEY in environment; "
                  "requests may fail without authentication")
        config["components"]["llm"]["model"]["config"]["base_url"] = "https://ollama.com"
        config["components"]["llm"]["model"]["config"]["headers"] = {
            "Authorization": f"Bearer {api_key or ''}"
        }
        print(f"Using Ollama cloud API (model: {args.model}, temperature: {args.temperature})")
    else:
        config["components"]["llm"]["model"]["config"]["base_url"] = "http://localhost:11434"
        # Remove headers if present from config template
        config["components"]["llm"]["model"]["config"].pop("headers", None)
        print(f"Using Ollama at localhost:11434 (model: {args.model})")

    print(f"Temperature: {args.temperature}")

    # Resolve input file path
    txt_file = str(ROOT_DIR / args.input)

    # Load pages
    if INPUT_MODE == "pagexml":
        print(f"Loading corrected PageXML from: {PAGEXML_DIR}")
        pages, line_offset_map = load_pages_from_pagexml(PAGEXML_DIR, SCAN_CSV)
    else:
        print(f"Loading text from: {txt_file}")
        if args.split_mode == "horizontal-rule":
            print(f"  Split mode: horizontal rules ('____________')")
            pages, line_offset_map = load_pages_from_txt_horizontal_rule(txt_file)
        else:
            print(f"  Split mode: bracket markers ([N])")
            pages, line_offset_map = load_pages_from_txt(txt_file)

    print(f"Loaded {len(pages)} pages")
    if not pages:
        print("ERROR: No pages were loaded from the input file.")
        print("  For text files with [N] page markers, use --split-mode brackets (default).")
        print("  For text files with horizontal-rule page separators, use --split-mode horizontal-rule.")
        print("  Alternatively, add [1], [2], ... markers at each page start in the source file.")
        return
    print(f"Starting {args.mode} NER with: {args.model} (language: {args.language})\n")

    # Assemble the Ollama LLM config as spacy nlp object
    nlp = assemble_from_config(config)
    print('Pipeline loaded:', nlp.pipe_names)

    # Monkey-patch LangChain's Ollama to print streaming tokens in real-time
    patch_langchain_ollama_streaming()

    # Collect all docs for merging later
    all_docs = []
    offset_map = {}  # pagenumber -> full-text character offset

    # Run NER per page
    sorted_pages = sorted(pages.items(), key=lambda x: _page_sort_key(x[0]))
    total_pages = len(sorted_pages)
    failed_pages = []
    t_start = time.perf_counter()
    for i, (pagenumber, (page_offset, page_text)) in enumerate(sorted_pages, 1):
        print(f"[{i}/{total_pages}] Processing page {pagenumber}...", end=" ", flush=True)

        # Unified retry loop: handles API failures (exceptions) and parsing failures (0 entities
        # despite LLM pipe-format response). Max 3 attempts total, with exponential
        # backoff (5s, 10s) between retries so server-side timeouts can recover.
        max_retries = 3
        doc = None
        for attempt in range(1, max_retries + 1):
            try:
                doc = nlp(page_text)
            except Exception as e:
                print(f"API ERROR (attempt {attempt}/{max_retries}: {e})", end="", flush=True)
                if attempt < max_retries:
                    delay = 5 * attempt  # 5s, 10s
                    print(f" — retrying in {delay}s...", flush=True)
                    time.sleep(delay)
                    continue
                print(" — giving up", flush=True)
                doc = nlp.make_doc(page_text)
                doc.user_data["error"] = str(e)
                failed_pages.append(pagenumber)
                break

            # Successful API call — check if entities were parsed
            num_ents = len(doc.ents)
            if num_ents > 0:
                print(f"done ({num_ents} entities)", flush=True)
                break
            else:
                # Check whether LLM attempted pipe format (parsing failure) or returned empty
                llm_io = doc.user_data.get("llm_io", {})
                llm_response = llm_io.get("llm", {}).get("response", None)
                if isinstance(llm_response, (list, tuple)):
                    llm_response = llm_response[0] if llm_response else ""
                attempted_format = " | True" in (llm_response or "") or " | False" in (llm_response or "")
                attempted_none_label = "==NONE==" in (llm_response or "")
                empty_response = not (llm_response or "").strip()

                if (attempted_format or attempted_none_label or empty_response) and attempt < max_retries:
                    reason = "empty response" if empty_response else "parsing failure"
                    print(f"0 entities ({reason}, retry {attempt}/{max_retries})...", flush=True)
                else:
                    if attempted_format or attempted_none_label or empty_response:
                        print(f"0 entities after {max_retries} attempts — giving up", flush=True)
                        failed_pages.append(pagenumber)
                    else:
                        print(f"done (0 entities)", flush=True)
                    break

        offset_map[pagenumber] = page_offset
        all_docs.append(doc)
    duration_seconds = round(time.perf_counter() - t_start, 1)

    # Save metadata (timing, config) alongside the spacy file
    meta_path = out_subdir / f"{letter_label}__{args.model}_t{args.temperature}_{args.mode}_{args.language}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump({
            "model": args.model,
            "mode": args.mode,
            "language": args.language,
            "temperature": args.temperature,
            "source_text": args.input,
            "duration_seconds": duration_seconds,
            "num_pages": total_pages,
            "ollama_host": args.ollama_host,
            "failed_pages": failed_pages,
        }, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    # Merge all docs into one DocBin and save
    merged_docbin = DocBin(docs=all_docs, store_user_data=True)
    spacy_path = out_subdir / f"{letter_label}__{args.model}_t{args.temperature}_{args.mode}_{args.language}.spacy"
    merged_docbin.to_disk(spacy_path)
    print(f"\nMerged DocBin saved to: {spacy_path}")

    # Save offset map for aligning with full-text annotations
    offset_map_path = out_subdir / f"{letter_label}__{args.model}_t{args.temperature}_{args.mode}_{args.language}_offset_map.json"
    with open(offset_map_path, 'w') as f:
        json.dump(offset_map, f)
    print(f"Offset map saved to: {offset_map_path}")

    # Save line offset map if available (PageXML mode)
    if line_offset_map is not None:
        line_map_path = out_subdir / f"{letter_label}__{args.model}_t{args.temperature}_{args.mode}_{args.language}_line_map.json"
        with open(line_map_path, 'w') as f:
            json.dump(line_offset_map, f)
        print(f"Line offset map saved to: {line_map_path}")


if __name__ == "__main__":
    main()

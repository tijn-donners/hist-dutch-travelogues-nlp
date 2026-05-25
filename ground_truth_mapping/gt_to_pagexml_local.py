import re
import time
import shutil
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import ollama
from difflib import SequenceMatcher

ROOT_DIR = Path(__file__).resolve().parent.parent

# Configuration
GT_FILE = str(ROOT_DIR / 'data/GT_1816_for_mapping.txt')
PAGE_DIR = str(ROOT_DIR / "data/page/")
OUTPUT_DIR = str(ROOT_DIR / "data/page_updated/")
MODEL_NAME = "gemma4:31b-cloud"  # Change to your preferred Ollama model
# os.environ["OLLAMA_HOST"] = "http://localhost:11434"  # If non‑default


# ----------------------------------------------------------------------
# Scan‑Number → Page‑Number mapping
# ----------------------------------------------------------------------
_scan_to_page_df = None   # Cache the CSV

def map_scan_to_page(scan_number):
    """
    Returns the page number (e.g. 'I', '22') for the given scan_number,
    or None if the scan is not in the mapping file.
    """
    global _scan_to_page_df
    if _scan_to_page_df is None:
        _scan_to_page_df = pd.read_csv(str(ROOT_DIR / 'data/1816-scannumber-to-pagenumber.csv'))
    
    result = _scan_to_page_df.loc[
        _scan_to_page_df['Scan Number'] == scan_number, 'Page Number'
    ]
    if result.empty:
        print(f"scan number: {scan_number} → NO MAPPING FOUND")
        return None

    page_number = result.item()
    print(f"scan number {scan_number} is mapped to pagenumber {page_number}")
    return page_number


def get_page_content(file_path, target_page):
    """
    Retrieve the ground truth transcription for target_page.
    Returns the content string if found, otherwise None.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    page_pattern = r'\[([IVXLCDM]+|\d+)\]'
    markers = list(re.finditer(page_pattern, text))
    
    if not markers:
        return None

    pages = {}
    for i in range(len(markers)):
        page_label = markers[i].group(1)
        start = markers[i].end()
        end = markers[i+1].start() if i + 1 < len(markers) else len(text)
        pages[page_label] = text[start:end].strip()

    return pages.get(target_page, None)


# ----------------------------------------------------------------------
# Page‑XML helpers
# ----------------------------------------------------------------------
def extract_scan_number(xml_filename):
    """Extract scan number from '0552_0179_0001.xml' → 1"""
    stem = Path(xml_filename).stem
    parts = stem.split('_')
    return int(parts[-1])


def _ns(root):
    """Retrieve namespace without string"""
    match = re.match(r'\{([^}]+)\}', root.tag)
    return match.group(1) if match else ''


def get_page_htr(xml_path):
    """Returns (full_text, line_ids, line_texts) from a PageXML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = _ns(root)

    ordered_group = root.find(f".//{{{ns}}}OrderedGroup")
    if ordered_group is None:
        return None, [], []

    lines_text = []
    line_ids = []
    for ref in ordered_group.findall(f"{{{ns}}}RegionRefIndexed"):
        region_id = ref.attrib['regionRef']
        region = root.find(f".//{{{ns}}}TextRegion[@id='{region_id}']")
        if region is not None:
            for line in region.findall(f"{{{ns}}}TextLine"):
                text_elem = line.find(f"{{{ns}}}TextEquiv/{{{ns}}}PlainText")
                text = text_elem.text if text_elem is not None and text_elem.text else ""
                lines_text.append(text)
                line_ids.append(line.attrib['id'])

    return " ".join(lines_text), line_ids, lines_text


def is_page_empty(xml_path):
    """Check if the PageXML contains any HTR text."""
    htr_text, _, _ = get_page_htr(str(xml_path))
    return htr_text is None or not htr_text.strip()


# ----------------------------------------------------------------------
# Ollama interaction
# ----------------------------------------------------------------------
def _parse_json_response(text):
    """Cleans Markdown code blocks from the AI response and parses it as JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()
    return json.loads(text)


def query_ollama(prompt):
    try:
        print(f"Prompting {MODEL_NAME}")
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={"temperature": 0},
            format="json"
        )
        return _parse_json_response(response.response)
    except Exception as e:
        print(f"  Error querying Ollama: {e}")
        return None


# ----------------------------------------------------------------------
# Alignment with confidence score
# ----------------------------------------------------------------------
def map_gt_to_lines(htr_lines, gt_text, page_label=None):
    """
    Corrects HTR lines using the page ground truth.
    Returns (line_mapping_dict, confidence) or (None, None).
    """
    page_hint = f" for page [{page_label}]" if page_label else ""
    numbered = "\n".join(f"[{i}] {line}" for i, line in enumerate(htr_lines))

    prompt = f'''
            You are an expert in historical Dutch text alignment.
            Below are HTR lines (noisy OCR) from a single page, and the ground truth text{page_hint}.
            === HTR lines ===
            {numbered}
            === Ground Truth text ===
            --- Content of Page {page_label} ---
            {gt_text}

            The HTR lines above define the line boundaries. For each [i] HTR line, find the
            contiguous portion of the Ground Truth text that corresponds to it on character level.
            Replace the entire HTR line with the the corresponding part of the Ground Truth.
            Together, the matched portions should reconstruct the Ground Truth text in order,
            without gaps or overlap (except for HTR lines that have no GT equivalent – page numbers,
            garbage, etc.).

            Return a JSON object with:
            - "lines": an object where keys are line numbers ("0", "1", …) and values are the
            **exact substring** of the Ground Truth for that line (keeping original spelling
            and punctuation).
            - "confidence": a number between 0.0 and 1.0 that reflects how confident you are
            that the overall alignment is correct (1.0 = perfect match for all lines,
            0.0 = completely unreliable).

            Rules:
            - Match by content order, NOT by exact text (HTR is noisy).
            - The HTR lines define where each line begins and ends — the GT tells you the
            correct spelling. Assign each HTR line only the GT text that belongs to it,
            not the entire remaining page.
            - If an HTR line has no GT equivalent (garbage, page number, etc.), omit it.
            - Use the exact spelling and punctuation from the Ground Truth!!!
            - HTR lines with only a few characters are often noise from another page → omit them.

            Example response:
            {{"lines": {{"0": "Personen", "1": "Inhoud der Brieven"}}, "confidence": 0.95}}
            '''
    result = query_ollama(prompt)
    if result is None:
        return None, None

    lines = result.get("lines") or result.get("mapping") or result
    mapping = {}
    for k, v in lines.items():
        try:
            mapping[int(k)] = v
        except (ValueError, TypeError):
            pass

    confidence = result.get("confidence", None)
    return mapping, confidence


def calculate_fuzzy_accuracy(original_gt, mapping):
    if not mapping:
        return 0.0
    
    recovered_text = "".join([mapping[k] for k in sorted(mapping.keys())])
    
    # ratio() returns a float from 0 to 1 based on how similar the strings are
    return SequenceMatcher(None, original_gt, recovered_text).ratio() * 100


# ----------------------------------------------------------------------
# Update PageXML
# ----------------------------------------------------------------------
def update_pagexml(xml_path, line_mapping, output_path=None):
    if not line_mapping:
        return
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = _ns(root)

    ordered_group = root.find(f".//{{{ns}}}OrderedGroup")
    if ordered_group is None:
        return

    idx = 0
    for ref in ordered_group.findall(f"{{{ns}}}RegionRefIndexed"):
        region_id = ref.attrib['regionRef']
        region = root.find(f".//{{{ns}}}TextRegion[@id='{region_id}']")
        if region is not None:
            for line in region.findall(f"{{{ns}}}TextLine"):
                if idx in line_mapping:
                    new_text = line_mapping[idx]
                    plain = line.find(f"{{{ns}}}TextEquiv/{{{ns}}}PlainText")
                    unicode_elem = line.find(f"{{{ns}}}TextEquiv/{{{ns}}}Unicode")
                    if plain is not None:
                        plain.text = new_text
                    if unicode_elem is not None:
                        unicode_elem.text = new_text
                idx += 1

    ET.register_namespace('', ns)
    write_path = output_path or xml_path
    tree.write(write_path, encoding='UTF-8', xml_declaration=True)


# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------
def main():
    xml_files = sorted(Path(PAGE_DIR).glob("*.xml"))
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    non_empty_files = [f for f in xml_files if not is_page_empty(f)]
    to_copy_as_is = len(xml_files) - len(non_empty_files)
    print(f"Found {len(xml_files)} PageXML files")
    print(f"  → {to_copy_as_is} empty pages (copied as‑is)")
    print(f"  → {len(non_empty_files)} pages to process\n")
    start_time = time.time()
    processed_count = 0
    page_stats = []
    detail_rows = []
    for xml_path in xml_files:
        out_path = output_dir / xml_path.name
        if is_page_empty(xml_path):
            shutil.copy2(str(xml_path), str(out_path))
            continue
        _, _, htr_lines = get_page_htr(str(xml_path))
        scan_number = extract_scan_number(xml_path.name)
        page_number = map_scan_to_page(scan_number)
        # --- Handle missing mapping ---
        if page_number is None:
            for idx, line_text in enumerate(htr_lines):
                detail_rows.append({
                    'filename': xml_path.name,
                    'line_index': idx,
                    'original_text': line_text,
                })
            page_stats.append({
                'filename': xml_path.name,
                'total_lines': len(htr_lines),
                'corrected_lines': 0,
                'unmatched_lines': len(htr_lines),
                'confidence': None,
                'fuzzy_accuracy': None,          # <-- NEW: add field
                'gt_chars_total': None,
                'gt_chars_mapped': None,
                'gt_coverage_pct': None,
                'status': 'no_mapping'
            })
            shutil.copy2(str(xml_path), str(out_path))
            processed_count += 1
            print(f"[{processed_count}/{len(non_empty_files)}] {xml_path.name}  "
                  f"SKIPPED (no page mapping)  "
                  f"({(time.time()-start_time)/60:.1f}m elapsed)")
            continue
        # --- Get ground truth (raw content only) ---
        gt_raw = get_page_content(GT_FILE, page_number)
        if gt_raw is None:
            # No GT content found for this page
            for idx, line_text in enumerate(htr_lines):
                detail_rows.append({
                    'filename': xml_path.name,
                    'line_index': idx,
                    'original_text': line_text,
                })
            page_stats.append({
                'filename': xml_path.name,
                'total_lines': len(htr_lines),
                'corrected_lines': 0,
                'unmatched_lines': len(htr_lines),
                'confidence': None,
                'fuzzy_accuracy': None,          # <-- NEW
                'gt_chars_total': None,
                'gt_chars_mapped': None,
                'gt_coverage_pct': None,
                'status': 'no_gt'
            })
            shutil.copy2(str(xml_path), str(out_path))
            processed_count += 1
            print(f"[{processed_count}/{len(non_empty_files)}] {xml_path.name}  "
                  f"SKIPPED (no GT for page {page_number})  "
                  f"({(time.time()-start_time)/60:.1f}m elapsed)")
            continue
        # Normal processing
        mapping, confidence = map_gt_to_lines(htr_lines, gt_raw, page_label=page_number)
        total_lines = len(htr_lines)
        corrected_count = len(mapping) if mapping else 0

        # --- Calculate fuzzy accuracy (the new part) ---
        if mapping is not None:
            fuzzy_accuracy = calculate_fuzzy_accuracy(gt_raw, mapping)
        else:
            fuzzy_accuracy = None

        # --- Ground truth coverage calculation ---
        total_gt_chars = len(gt_raw)
        if mapping:
            mapped_gt_chars = sum(len(v) for v in mapping.values())
        else:
            mapped_gt_chars = 0
        coverage = (mapped_gt_chars / total_gt_chars * 100) if total_gt_chars > 0 else None

        # Progress line (add fuzzy accuracy)
        processed_count += 1
        elapsed = time.time() - start_time
        remaining_count = len(non_empty_files) - processed_count
        if processed_count > 1:
            per_item = elapsed / processed_count
            eta_sec = per_item * remaining_count
            eta = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
        else:
            eta = "?"
        conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
        fuzzy_str = f"{fuzzy_accuracy:.1f}%" if fuzzy_accuracy is not None else "N/A"
        cov_str = f"{coverage:.1f}%" if coverage is not None else "N/A"
        print(f"[{processed_count}/{len(non_empty_files)}] {xml_path.name}  "
              f"{corrected_count}/{total_lines} lines  conf={conf_str}  "
              f"fuzz={fuzzy_str}  GT cov={cov_str}  "   # <-- added fuzz=
              f"({elapsed/60:.1f}m elapsed, ~{eta} remaining)")

        # Unmatched lines detail
        for idx, line_text in enumerate(htr_lines):
            matched = mapping is not None and idx in mapping
            if not matched:
                detail_rows.append({
                    'filename': xml_path.name,
                    'line_index': idx,
                    'original_text': line_text,
                })
        page_stats.append({
            'filename': xml_path.name,
            'total_lines': total_lines,
            'corrected_lines': corrected_count,
            'unmatched_lines': total_lines - corrected_count,
            'confidence': confidence,
            'fuzzy_accuracy': fuzzy_accuracy,      # <-- NEW
            'gt_chars_total': total_gt_chars,
            'gt_chars_mapped': mapped_gt_chars,
            'gt_coverage_pct': coverage,
            'status': 'mapping_attempted'
        })
        if mapping:
            update_pagexml(str(xml_path), mapping, str(out_path))
            print(f"  ✓ Updated ({corrected_count} lines)")
        else:
            print(f"  ✗ No mapping returned – copied as‑is")
            shutil.copy2(str(xml_path), str(out_path))

    elapsed_total = time.time() - start_time
    print(f"\nDone in {elapsed_total/60:.1f}m. Files written to {OUTPUT_DIR}")
    # Summary report
    if page_stats:
        df_summary = pd.DataFrame(page_stats)
        summary_path = output_dir / "report_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"\nSummary report saved to {summary_path}")
        total_corrected = df_summary['corrected_lines'].sum()
        total_lines_all = df_summary['total_lines'].sum()
        total_unmatched = df_summary['unmatched_lines'].sum()
        print(f"  Total: {total_corrected}/{total_lines_all} lines corrected "
              f"({total_corrected/total_lines_all*100:.1f}%)")
        print(f"  Unmatched lines (see detail report): {total_unmatched}")
        valid_conf = df_summary['confidence'].dropna()
        if not valid_conf.empty:
            print(f"  Average confidence (over mapped pages): {valid_conf.mean():.2f}")
        # --- Average fuzzy accuracy ---
        valid_fuzz = df_summary['fuzzy_accuracy'].dropna()       # <-- NEW
        if not valid_fuzz.empty:
            print(f"  Average fuzzy accuracy (over mapped pages): {valid_fuzz.mean():.1f}%")
        # Aggregate ground truth coverage
        gt_total_all = df_summary['gt_chars_total'].dropna().sum()
        gt_mapped_all = df_summary['gt_chars_mapped'].dropna().sum()
        if gt_total_all > 0:
            overall_cov = gt_mapped_all / gt_total_all * 100
            print(f"  Overall GT coverage: {overall_cov:.2f}% "
                  f"({int(gt_mapped_all)}/{int(gt_total_all)} characters)")
        no_map_count = (df_summary['status'] == 'no_mapping').sum()
        no_gt_count = (df_summary['status'] == 'no_gt').sum()
        if no_map_count or no_gt_count:
            print(f"  Pages skipped: {no_map_count} (no mapping), {no_gt_count} (no GT)")
    if detail_rows:
        df_detail = pd.DataFrame(detail_rows)
        detail_path = output_dir / "report_unmatched.csv"
        df_detail.to_csv(detail_path, index=False)
        print(f"Unmatched lines saved to {detail_path}")
        print(f"  ({len(detail_rows)} lines to review manually)")

if __name__ == "__main__":
    main()
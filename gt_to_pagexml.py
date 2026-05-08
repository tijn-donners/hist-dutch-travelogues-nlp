import re
import time
import shutil
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import ollama

# Configuration
GT_FILE = "/home/tijn-do/hist-dutch-travelogues-nlp/data/GT_1816_processed.txt"
PAGE_DIR = "/home/tijn-do/hist-dutch-travelogues-nlp/data/page/"
OUTPUT_DIR = "/home/tijn-do/hist-dutch-travelogues-nlp/data/page_updated/"
MODEL_NAME = "gemma4:31b"  # Change this to the preferred model available in Ollama

# Point to your Ollama server if not the default (http://localhost:11434)
# os.environ["OLLAMA_HOST"] = "http://localhost:11434"


def read_gt():
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    pages = content.split('TRP_PAGEBREAK')
    pages = [p for p in pages if p.strip()]
    return pages


def _ns(root):
    match = re.match(r'\{([^}]+)\}', root.tag)
    return match.group(1) if match else ''


def get_page_htr(xml_path):
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
                text = line.find(f"{{{ns}}}TextEquiv/{{{ns}}}PlainText").text or ""
                lines_text.append(text)
                line_ids.append(line.attrib['id'])

    return " ".join(lines_text), line_ids, lines_text


def _parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()
    return json.loads(text)


def query_ollama(prompt):
    try:
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


def map_gt_to_lines(htr_lines, labeled_context):
    """Ask the LLM to correct each HTR line using the ground truth.

    htr_lines: list of strings, one per TextLine element
    labeled_context: string with labeled GT segments (Segment 0, Segment 1, ...)

    Returns: (line_mapping dict, set of used segment indices) or (None, None)
    """
    numbered = "\n".join(f"[{i}] {line}" for i, line in enumerate(htr_lines))
    prompt = f'''
You are an expert in historical Dutch text alignment.
Below are HTR lines (noisy OCR) from a single page, and ground truth segments.

=== HTR lines ===
{numbered}

=== Ground Truth segments ===
{labeled_context}

For each [i] HTR line, find its corrected text in the Ground Truth segments.
Return a JSON object with two fields:
- "lines": an object where keys are line numbers ("0", "1", ...) and values are the corrected text
- "segments": a list of segment numbers that were used (e.g. [0] or [0, 1] or [2])

Rules:
- Match by content order, NOT by exact text (HTR is noisy)
- If an HTR line has no GT equivalent (garbage, page number, etc.), skip it
- Use the exact spelling and punctuation from the Ground Truth
- Report which Segment number(s) the matches came from

Example:
{{"lines": {{"0": "Personen", "1": "Inhoud der Brieven"}}, "segments": [0]}}
'''
    result = query_ollama(prompt)
    if result is None:
        return None, None

    lines = result.get("lines") or result.get("mapping") or result
    segments_used = set(result.get("segments", []))

    # Parse line numbers -> corrected text
    mapping = {}
    for k, v in lines.items():
        try:
            mapping[int(k)] = v
        except (ValueError, TypeError):
            pass

    return mapping, segments_used


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


def is_page_empty(xml_path):
    htr_text, _, _ = get_page_htr(str(xml_path))
    return htr_text is None or not htr_text.strip()


def main():
    gt_pages = read_gt()
    xml_files = sorted([f for f in Path(PAGE_DIR).glob("*.xml")])

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-scan
    non_empty_files = [f for f in xml_files if not is_page_empty(f)]
    to_copy_as_is = len(xml_files) - len(non_empty_files)
    print(f"Found {len(xml_files)} PageXML files, {len(gt_pages)} GT segments")
    print(f"  → {to_copy_as_is} empty pages (will be copied as-is)")
    print(f"  → {len(non_empty_files)} pages to process\n")

    gt_idx = 0
    start_time = time.time()
    processed_count = 0

    # Stats for the final report
    page_stats = []
    detail_rows = []

    for xml_path in xml_files:
        out_path = output_dir / xml_path.name

        if is_page_empty(xml_path):
            shutil.copy2(str(xml_path), str(out_path))
            continue

        if gt_idx < len(gt_pages):
            _, _, htr_lines = get_page_htr(str(xml_path))

            # Build a window of labeled GT segments (current + next 4)
            gt_window_end = min(gt_idx + 5, len(gt_pages))
            labeled_segments = []
            for i, seg in enumerate(gt_pages[gt_idx:gt_window_end]):
                labeled_segments.append(f"--- Segment {i} ---\n{seg}")
            labeled_context = "\n\n".join(labeled_segments)

            # Get mapping with segment info
            mapping, segments_used = map_gt_to_lines(htr_lines, labeled_context)

            total_lines = len(htr_lines)
            corrected_count = len(mapping) if mapping else 0

            # Progress
            processed_count += 1
            elapsed = time.time() - start_time
            done = processed_count
            remaining_count = len(non_empty_files) - processed_count
            if done > 1:
                per_item = elapsed / done
                remaining = per_item * remaining_count
                eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
            else:
                eta = "?"
            print(f"[{done}/{len(non_empty_files)}] {xml_path.name}  "
                  f"{corrected_count}/{total_lines} lines  "
                  f"({elapsed/60:.1f}m elapsed, ~{eta} remaining)")

            # Track unmatched lines for review
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
            })

            if mapping:
                update_pagexml(str(xml_path), mapping, str(out_path))

                if segments_used:
                    first_segment = min(segments_used)
                    advance = first_segment + 1
                    if advance > 1:
                        print(f"  ✓ Updated ({corrected_count} lines, segment(s) "
                              f"{segments_used} — advancing gt_idx by {advance})")
                    else:
                        print(f"  ✓ Updated ({corrected_count} lines)")
                else:
                    advance = 1
                    print(f"  ✓ Updated ({corrected_count} lines)")

                gt_idx += advance
            else:
                print(f"  ✗ No mapping found, copied as-is")
                shutil.copy2(str(xml_path), str(out_path))

        else:
            shutil.copy2(str(xml_path), str(out_path))

    elapsed_total = time.time() - start_time
    print(f"\nDone in {elapsed_total/60:.1f}m. Files written to {OUTPUT_DIR}")
    print(f"Final GT segment index: {gt_idx} / {len(gt_pages)}")

    # Generate CSV reports
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

    if detail_rows:
        df_detail = pd.DataFrame(detail_rows)
        detail_path = output_dir / "report_unmatched.csv"
        df_detail.to_csv(detail_path, index=False)
        print(f"Unmatched lines saved to {detail_path}")
        print(f"  ({len(detail_rows)} lines to review manually)")


if __name__ == "__main__":
    main()

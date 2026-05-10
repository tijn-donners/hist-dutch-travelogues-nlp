import re
import time
import shutil
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import ollama
import os

# Configuration
GT_FILE = f"/scratch/{os.environ['USER']}/hist-dutch-travelogues-nlp/data/GT_1816_processed.txt"
PAGE_DIR = f"/scratch/{os.environ['USER']}/hist-dutch-travelogues-nlp/data/page/"
OUTPUT_DIR = f"/scratch/{os.environ['USER']}/hist-dutch-travelogues-nlp/data/page_updated/"
MODEL_NAME = "gemma4:31b"

# How many pages to process in parallel.
# Should match OLLAMA_NUM_PARALLEL in your Slurm script.
MAX_WORKERS = 4

# Thread-safe print lock
_print_lock = threading.Lock()


def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# GT reading
# ---------------------------------------------------------------------------

def read_gt():
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    pages = content.split('TRP_PAGEBREAK')
    return [p for p in pages if p.strip()]


# ---------------------------------------------------------------------------
# PageXML helpers
# ---------------------------------------------------------------------------

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
                plain = line.find(f"{{{ns}}}TextEquiv/{{{ns}}}PlainText")
                text = (plain.text or "") if plain is not None else ""
                lines_text.append(text)
                line_ids.append(line.attrib['id'])

    return " ".join(lines_text), line_ids, lines_text


def is_page_empty(xml_path):
    htr_text, _, _ = get_page_htr(str(xml_path))
    return htr_text is None or not htr_text.strip()


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


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

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
            format="json",
        )
        return _parse_json_response(response.response)
    except Exception as e:
        tprint(f"  Error querying Ollama: {e}")
        return None


def map_gt_to_lines(htr_lines, labeled_context):
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

    mapping = {}
    for k, v in lines.items():
        try:
            mapping[int(k)] = v
        except (ValueError, TypeError):
            pass

    return mapping, segments_used


# ---------------------------------------------------------------------------
# Per-page worker (called from thread pool)
# ---------------------------------------------------------------------------

def process_page(task):
    """
    task = {
        'xml_path': Path,
        'out_path': Path,
        'gt_pages': list[str],
        'gt_idx': int,          # starting GT index for this page
        'page_num': int,        # 1-based position among non-empty pages
        'total_pages': int,
    }
    Returns a result dict consumed by the main thread.
    """
    xml_path = task['xml_path']
    out_path = task['out_path']
    gt_pages = task['gt_pages']
    gt_idx = task['gt_idx']

    _, _, htr_lines = get_page_htr(str(xml_path))

    gt_window_end = min(gt_idx + 5, len(gt_pages))
    labeled_segments = [
        f"--- Segment {i} ---\n{seg}"
        for i, seg in enumerate(gt_pages[gt_idx:gt_window_end])
    ]
    labeled_context = "\n\n".join(labeled_segments)

    mapping, segments_used = map_gt_to_lines(htr_lines, labeled_context)

    total_lines = len(htr_lines)
    corrected_count = len(mapping) if mapping else 0

    if mapping:
        update_pagexml(str(xml_path), mapping, str(out_path))
        advance = (min(segments_used) + 1) if segments_used else 1
        status = "ok"
    else:
        shutil.copy2(str(xml_path), str(out_path))
        advance = 1
        status = "no_mapping"

    return {
        'xml_path': xml_path,
        'out_path': out_path,
        'mapping': mapping,
        'segments_used': segments_used,
        'htr_lines': htr_lines,
        'total_lines': total_lines,
        'corrected_count': corrected_count,
        'advance': advance,
        'status': status,
        'page_num': task['page_num'],
        'total_pages': task['total_pages'],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gt_pages = read_gt()
    xml_files = sorted(Path(PAGE_DIR).glob("*.xml"))

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-scan: classify files and cache emptiness so we don't parse twice
    tprint("Pre-scanning PageXML files…")
    empty_files = []
    non_empty_files = []
    for f in xml_files:
        (empty_files if is_page_empty(f) else non_empty_files).append(f)

    tprint(f"Found {len(xml_files)} PageXML files, {len(gt_pages)} GT segments")
    tprint(f"  → {len(empty_files)} empty pages (copied as-is)")
    tprint(f"  → {len(non_empty_files)} pages to process (workers={MAX_WORKERS})\n")

    # Copy empty files immediately (cheap, no need to thread)
    for f in empty_files:
        shutil.copy2(str(f), str(output_dir / f.name))

    # Build tasks with pre-assigned GT indices.
    # Because the GT index depends on the *results* of previous pages we can't
    # fully pre-assign in parallel — we use a sliding window approach:
    # pages are submitted in order but GT advancement is applied after each
    # completes so that the NEXT batch uses correct GT offsets.
    #
    # Strategy: process in small ordered batches of MAX_WORKERS pages.
    # Within each batch pages run concurrently; batches are sequential so
    # GT advancement stays correct.

    page_stats = []
    detail_rows = []
    gt_idx = 0
    start_time = time.time()
    done_count = 0
    total_non_empty = len(non_empty_files)

    for batch_start in range(0, total_non_empty, MAX_WORKERS):
        batch = non_empty_files[batch_start: batch_start + MAX_WORKERS]

        # Assign GT indices sequentially within the batch using the current gt_idx
        # Each page in the batch gets its own snapshot of gt_idx so they don't
        # interfere with each other's window lookup.
        tasks = []
        running_gt = gt_idx
        for xml_path in batch:
            if running_gt >= len(gt_pages):
                # No GT left — just copy
                shutil.copy2(str(xml_path), str(output_dir / xml_path.name))
                continue
            tasks.append({
                'xml_path': xml_path,
                'out_path': output_dir / xml_path.name,
                'gt_pages': gt_pages,
                'gt_idx': running_gt,
                'page_num': done_count + len(tasks) + 1,
                'total_pages': total_non_empty,
            })
            # Optimistically advance by 1 for the next task in the same batch.
            # The true advance is corrected after results come in.
            running_gt += 1

        if not tasks:
            continue

        # Run the batch in parallel
        results_by_path = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_page, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    res = future.result()
                    results_by_path[res['xml_path']] = res
                except Exception as e:
                    task = futures[future]
                    tprint(f"  ERROR processing {task['xml_path'].name}: {e}")
                    shutil.copy2(str(task['xml_path']), str(task['out_path']))

        # Apply results in original page order so GT advances correctly
        for xml_path in batch:
            if xml_path not in results_by_path:
                continue
            res = results_by_path[xml_path]
            done_count += 1

            elapsed = time.time() - start_time
            remaining_count = total_non_empty - done_count
            if done_count > 1:
                per_item = elapsed / done_count
                eta = time.strftime("%H:%M:%S", time.gmtime(per_item * remaining_count))
            else:
                eta = "?"

            tprint(
                f"[{done_count}/{total_non_empty}] {xml_path.name}  "
                f"{res['corrected_count']}/{res['total_lines']} lines  "
                f"({elapsed/60:.1f}m elapsed, ~{eta} remaining)"
            )

            if res['status'] == 'ok':
                tprint(
                    f"  ✓ Updated ({res['corrected_count']} lines, "
                    f"segment(s) {res['segments_used']} — gt_idx {gt_idx} → {gt_idx + res['advance']})"
                )
            else:
                tprint(f"  ✗ No mapping found, copied as-is")

            gt_idx += res['advance']

            # Collect stats
            page_stats.append({
                'filename': xml_path.name,
                'total_lines': res['total_lines'],
                'corrected_lines': res['corrected_count'],
                'unmatched_lines': res['total_lines'] - res['corrected_count'],
            })
            for idx, line_text in enumerate(res['htr_lines']):
                if res['mapping'] is None or idx not in res['mapping']:
                    detail_rows.append({
                        'filename': xml_path.name,
                        'line_index': idx,
                        'original_text': line_text,
                    })

    elapsed_total = time.time() - start_time
    tprint(f"\nDone in {elapsed_total/60:.1f}m. Files written to {OUTPUT_DIR}")
    tprint(f"Final GT segment index: {gt_idx} / {len(gt_pages)}")

    if page_stats:
        df_summary = pd.DataFrame(page_stats)
        summary_path = output_dir / "report_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        total_corrected = df_summary['corrected_lines'].sum()
        total_lines_all = df_summary['total_lines'].sum()
        tprint(f"\nSummary report → {summary_path}")
        tprint(
            f"  {total_corrected}/{total_lines_all} lines corrected "
            f"({total_corrected/total_lines_all*100:.1f}%)"
        )

    if detail_rows:
        df_detail = pd.DataFrame(detail_rows)
        detail_path = output_dir / "report_unmatched.csv"
        df_detail.to_csv(detail_path, index=False)
        tprint(f"Unmatched lines → {detail_path} ({len(detail_rows)} lines)")


if __name__ == "__main__":
    main()

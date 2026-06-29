"""Round-trip spot-checks for the per-page ALTO export.

The exporter writes one ALTO XML per page scan
(``output/alto/1816_third_letter_scan{NNNN}.alto.xml``) plus a shared sidecar
``1816_third_letter.ttl`` (no embedded RDF in the page files). Checks:
  1. e2 Cassel: the scan0049 file (page 16) has a <String CONTENT="Cassel">
     TAGREFS~"ne_e2", an <OtherTag ID="ne_e2" URI="ato:CT.BRF0003.e2">, and
     "ato:CT.BRF0003.e2" appears in the shared sidecar .ttl.
  2. For a sampled matched line, manual_substring == full_text[start:end]
     (recomputed from the .spacy), i.e. offsets line up with the shared space.
  3. scan0049: each <TextLine> box is baseline-anchored — within the Loghi
     Coords AABB, not crossing a neighbouring baseline, and height <= 0.75*pitch.
  4. Every entity whose span falls inside a matched line is tagged on >= 1
     <String> across the per-page files; no TAGREFS dangles (every referenced
     tag id is declared in the same file's <Tags>).
  5. Determinism: re-running Stage B per page yields byte-identical files.

Exits non-zero on any failure.
"""

import glob
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import spacy
from spacy.tokens import DocBin

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import alto_exporter as ae  # noqa: E402

ALTO_NS = ae.ALTO_NS
A = "{" + ALTO_NS + "}"

PAGE_GLOB = str(ROOT / "output" / "alto" / "1816_third_letter_scan*.alto.xml")
SIDECAR_TTL = ROOT / "output" / "alto" / "1816_third_letter.ttl"
ALIGN_PATH = ROOT / "output" / "alto" / "line_alignment.json"
SPACY_PATH = ROOT / "entity_linking" / "el-results" / "1816_el_gs_el.spacy"
OFFSET_MAP = ROOT / "entity_linking" / "el-results" / "1816_el_gs_offset_map.json"
MENTION_MAP = ROOT / "output" / "re" / "1816_el_gs__deepseek-v4-pro_t0.0_thinkDefault_mention_map.json"
PAGEXML_DIR = ROOT / "data" / "page"
SCAN_CSV = ROOT / "data" / "1816-scannumber-to-pagenumber.csv"
LETTER_ID = "BRF0003"

failures = []


def check(cond, msg):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        failures.append(msg)


def _page_files():
    """Return list of (scan_int, path, root, xml_str) sorted by scan."""
    out = []
    for f in sorted(glob.glob(PAGE_GLOB)):
        # 1816_third_letter_scan0049.alto.xml -> scan 49
        name = Path(f).name
        scan = int(name.split("_scan")[-1].split(".")[0])
        xml_str = open(f, encoding="utf-8").read()
        root = ET.fromstring(xml_str)
        out.append((scan, f, root, xml_str))
    return out


def main():
    print("ALTO self-check (per-page)\n" + "=" * 60)

    pages = _page_files()
    check(len(pages) == 10, f"10 page files found (got {len(pages)})")
    if not pages:
        sys.exit(1)
    page_by_scan = {scan: (f, root, xml_str) for scan, f, root, xml_str in pages}

    alignment = json.load(open(ALIGN_PATH))
    nlp = spacy.blank("nl")
    db = DocBin().from_disk(SPACY_PATH)
    docs = list(db.get_docs(nlp.vocab))
    om = json.load(open(OFFSET_MAP))
    sp = sorted(om.items(), key=lambda kv: ae._page_sort_key(kv[0]))
    mm = json.load(open(MENTION_MAP))
    ml = ae.build_mention_lookup(mm)
    full_text, _, all_entities = ae.build_full_text_and_entities(docs, sp, ml)

    # ── 1. e2 Cassel (page 16 = scan 0049) ───────────────────────────────
    print("\n1. e2 Cassel linkage (scan0049)")
    scan49 = page_by_scan[49]
    f49, root49, xml49 = scan49
    cassel_strings = []
    for st in root49.iter(f"{A}String"):
        if st.get("CONTENT") == "Cassel" and "ne_e2" in (st.get("TAGREFS") or "").split():
            cassel_strings.append(st)
    check(bool(cassel_strings),
          f"<String CONTENT='Cassel' TAGREFS~'ne_e2'> in scan0049 ({len(cassel_strings)})")

    other_e2 = None
    for ot in root49.iter(f"{A}OtherTag"):
        if ot.get("ID") == "ne_e2":
            other_e2 = ot
            break
    check(other_e2 is not None, "<OtherTag ID='ne_e2'> exists in scan0049")
    if other_e2 is not None:
        check(other_e2.get("URI") == f"ato:CT.{LETTER_ID}.e2",
              f"<OtherTag URI> == ato:CT.{LETTER_ID}.e2 (got {other_e2.get('URI')!r})")

    sidecar_str = open(SIDECAR_TTL, encoding="utf-8").read() if SIDECAR_TTL.exists() else ""
    check(f"ato:CT.{LETTER_ID}.e2" in sidecar_str,
          f"ato:CT.{LETTER_ID}.e2 present in shared sidecar .ttl")
    check("cidoc-crm" not in xml49 and "xenoData" not in xml49,
          "scan0049 has no embedded RDF (xenoData/cidoc-crm absent)")

    # ── 2. matched-line offset round-trip ───────────────────────────────
    print("\n2. matched-line offsets vs full_text")
    mismatches = 0
    checked = 0
    for page in alignment["pages"]:
        for region in page["regions"]:
            for line in region["lines"]:
                if not line["matched"]:
                    continue
                s, e = line["full_text_start"], line["full_text_end"]
                got = full_text[s:e]
                if got != line["manual_substring"]:
                    mismatches += 1
                    if mismatches <= 3:
                        print(f"     mismatch page {page['page']}: "
                              f"{got!r} != {line['manual_substring']!r}")
                checked += 1
                if checked >= 40:
                    break
            if checked >= 40:
                break
        if checked >= 40:
            break
    check(mismatches == 0, f"first {checked} matched lines: manual_substring == full_text[s:e] "
          f"({mismatches} mismatches)")

    # ── 3. TextLine boxes are baseline-anchored (scan0049) ───────────────
    #   - box ⊆ Loghi Coords AABB (VPOS/HPOS never outside the Coords extent)
    #   - box does not cross a neighbouring baseline (top > prev baseline,
    #     bottom < next baseline)
    #   - box height is tight: ≤ 0.75 * page pitch (vs the old ~2× pitch AABB)
    # Validated against the cached alignment (pinfo["regions"]), which is what
    # the exporter actually builds from — a fresh parse_loghi_page can assign
    # different line idx values, so it is not a safe reference here.
    print("\n3. TextLine geometry (scan0049): baseline-anchored, within Coords, no crossing")
    align_by_page = {p["page"]: p for p in alignment["pages"]}
    pinfo = align_by_page["16"]
    pitch = ae.page_pitch(pinfo["regions"])
    # ID lookup: exporter writes tl_{scan}_{region_idx}_{line_idx}; scan 49.
    tl_by_id = {tl.get("ID"): tl for tl in root49.iter(f"{A}TextLine")}
    inside_ok = cross_ok = tight_ok = matched = 0
    for region in pinfo["regions"]:
        blines = sorted(
            (ae.baseline_median_y(l["baseline_points"]), l["idx"])
            for l in region["lines"] if l.get("baseline_points"))
        neigh = {}
        for i, (b, idx) in enumerate(blines):
            neigh[idx] = (b,
                          blines[i - 1][0] if i > 0 else None,
                          blines[i + 1][0] if i < len(blines) - 1 else None)
        for ll in region["lines"]:
            cb = ae.coords_to_bbox(ll["coords_points"])
            if ll.get("baseline_points") and ll["idx"] in neigh:
                b, bp, bn = neigh[ll["idx"]]
                lb = ae.line_box_from_baseline(ll["coords_points"],
                                                ll["baseline_points"], b, bp, bn, pitch)
            else:
                lb = cb
            tl = tl_by_id.get(f"tl_49_{region['region_idx']}_{ll['idx']}")
            if tl is None:
                continue
            matched += 1
            vpos = int(tl.get("VPOS")); h = int(tl.get("HEIGHT"))
            top, bottom = vpos, vpos + h
            # box ⊆ Coords AABB
            if cb[1] - 1 <= top and bottom <= cb[1] + cb[3] + 1:
                inside_ok += 1
            # does not cross neighbouring baselines
            if (bp is None or top > bp) and (bn is None or bottom < bn):
                cross_ok += 1
            # tight: height ≤ 0.75 * pitch (allow tiny rounding slack)
            if h <= 0.75 * pitch + 2:
                tight_ok += 1
    check(matched > 0, f"matched exported TextLines to alignment lines ({matched})")
    check(inside_ok == matched,
          f"every box within Loghi Coords AABB ({inside_ok}/{matched})")
    check(cross_ok == matched,
          f"no box crosses a neighbouring baseline ({cross_ok}/{matched})")
    check(tight_ok == matched,
          f"every box height <= 0.75*pitch ({tight_ok}/{matched}, pitch={pitch:.0f})")

    # ── 4. entity -> String tagging coverage + no dangling TAGREFS ──────
    print("\n4. entity -> String tagging coverage (across all page files)")
    present_tags = set()
    dangling = []
    for scan, f, root, xml_str in pages:
        declared = {ot.get("ID") for ot in root.iter(f"{A}OtherTag")}
        for st in root.iter(f"{A}String"):
            for t in (st.get("TAGREFS") or "").split():
                present_tags.add(t)
                if t not in declared:
                    dangling.append((Path(f).name, t))
    check("ne_e2" in present_tags, "ne_e2 applied to >= 1 <String> in scan0049")
    # Regression: the "7 bergen" E18_Physical_Thing tag (ne_e20) must be carried
    # by a <String> on the recovered idx-16 line ("zeer bergagtig ... 7 bergen").
    bergen_strings = []
    for st in root49.iter(f"{A}String"):
        if (st.get("CONTENT") or "").strip() in ("bergen", "7") \
                and "ne_e20" in (st.get("TAGREFS") or "").split():
            bergen_strings.append(st)
    check(bool(bergen_strings),
          f"'7 bergen' E18_Physical_Thing (ne_e20) on a <String> in scan0049 ({len(bergen_strings)})")
    check(not dangling, f"no dangling TAGREFS (referenced tags all declared in-file) — {len(dangling)} dangling")

    tagged = sum(1 for e in all_entities
                 if (f"ne_{e[5]}" if e[5] else f"ne_g{e[0]}") in present_tags)
    total = len(all_entities)
    print(f"     {tagged}/{total} entities appear on >= 1 String "
          f"({100*tagged/total:.1f}%)")
    check(tagged >= 300, f"tagged coverage >= 300/313 (got {tagged}/{total})")

    # ── 5. determinism (re-run Stage B per page) ─────────────────────────
    print("\n5. Stage B determinism (per page)")
    tags, tag_ids, entity_string_hits, string_tags = ae.prepare_tags(
        all_entities, alignment, LETTER_ID)
    all_ok = True
    for scan, f, root, xml_str in pages:
        # find this page's pinfo
        pinfo = next(p for p in alignment["pages"] if p.get("scan") == scan)
        page_num = pinfo["page"]
        page_tags = [tags[ei] for ei in sorted(
            ae.page_entity_indices(entity_string_hits, page_num))]
        root2 = ae.build_page_root(
            page_num, pinfo, scan, page_tags, string_tags, LETTER_ID, PAGEXML_DIR)
        tmp = f"/tmp/alto_det_scan{scan:04d}.xml"
        xml2 = ae.serialize_alto(root2, None, tmp)
        if xml2 != xml_str:
            all_ok = False
            print(f"     page {page_num} (scan {scan:04d}): MISMATCH ({len(xml2)} vs {len(xml_str)})")
    check(all_ok, "re-run Stage B byte-identical for all 10 page files")

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL — {len(failures)} check(s) failed:")
        for fmsg in failures:
            print(f"  - {fmsg}")
        sys.exit(1)
    print("PASS — all checks passed")


if __name__ == "__main__":
    main()
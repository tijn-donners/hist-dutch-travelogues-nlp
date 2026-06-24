"""Create a CSV template from Recogito ground-truth annotations for EL evaluation.

Reads `ner/gs-annotations.jsonld`, extracts every annotated entity with its CIDOC
type, character offsets, and *pre-fills* Wikidata QID and GeoNames ID where they
exist in the Recogito comment bodies. Remaining blanks can be filled manually.

The CSV serves double duty:
  1. End-to-end NER+EL evaluation — compare pipeline NER spans + EL IDs
     against the gold-standard spans and IDs in this CSV.
  2. EL-only evaluation — given the gold-standard entity spans, check whether
     the Entity Linking module predicts the correct IDs.
"""

import json
import re
from pathlib import Path

import pandas as pd

# Paths
GS_FILE = Path(__file__).resolve().parent / "gs_annotations" / "1816.jsonld"

# CIDOC CRM mapping: Recogito body ID → pipeline label
TYPE_MAP = {
    "E53": "E53_Place",
    "E18": "E18_Physical_Thing",
    "E52": "E52_Time_Span",
    "E31": "E31_Document",
    "E19": "E19_Physical_Object",
    "E20": "E20_Biological_Object",
    "F2": "F2_Expression",
}

# Value-based mapping: some Recogito tagging bodies have the type only in the
# "value" field (no CIDOC CRM URI in the "id" field), e.g. Mode of Transportation.
VALUE_MAP = {
    "E53 Place": "E53_Place",
    "E18 Physical Thing": "E18_Physical_Thing",
    "E52 Time-Span": "E52_Time_Span",
    "E31 Document": "E31_Document",
    "E19 Physical Object": "E19_Physical_Object",
    "E20 Biological Object": "E20_Biological_Object",
    "Mode of Transportation": "Mode_of_Transportation",
    "F2 Expression": "F2_Expression",
}

# ── URL patterns in Recogito comment bodies ──────────────────────────────────
WIKIDATA_RE = re.compile(
    r"(?:https?://)?(?:www\.)?wikidata\.org/(?:entity|wiki)/(Q\d+)",
    re.IGNORECASE,
)
GEONAMES_RE = re.compile(
    r"(?:https?://)?(?:www\.)?geonames\.org/(\d+)",
    re.IGNORECASE,
)


def extract_ids_from_comment(value: str):
    """Parse Wikidata QID and GeoNames ID from a Recogito comment body."""
    wd_match = WIKIDATA_RE.search(value)
    gn_match = GEONAMES_RE.search(value)
    return (wd_match.group(1) if wd_match else None,
            gn_match.group(1) if gn_match else None)


def get_cidoc_type(body_list):
    """Return the pipeline label from the tagging body.

    Checks both the 'id' field (CIDOC CRM URI, e.g. E53, E18) and the
    'value' field (human-readable label, e.g. "Mode of Transportation").
    """
    for b in body_list:
        if b.get("purpose") != "tagging":
            continue
        # Check 'id' field for CIDOC CRM URI
        bid = b.get("id", "")
        for cidoc_key, pipeline_label in TYPE_MAP.items():
            if cidoc_key in bid:
                return pipeline_label
        # Check 'value' field for types without a CIDOC CRM URI
        val = b.get("value", "")
        if val in VALUE_MAP:
            return VALUE_MAP[val]
    return None


def get_comment_text(body_list):
    """Return the value of the 'commenting' body, if any."""
    for b in body_list:
        if b.get("purpose") == "commenting":
            return b.get("value", "")
    return ""


def main():
    if not GS_FILE.exists():
        print(f"Ground truth file not found: {GS_FILE}")
        raise SystemExit(1)

    with open(GS_FILE, encoding="utf-8") as f:
        annotations = json.load(f)

    rows = []
    seen = set()

    for ann in annotations:
        # --- Entity type ---
        label = get_cidoc_type(ann.get("body", []))
        if label is None:
            continue

        # --- Offsets and text ---
        target = ann.get("target", {})
        selectors = target.get("selector", [])
        pos_sel = None
        quote_sel = None
        for s in selectors:
            if s.get("type") == "TextPositionSelector":
                pos_sel = s
            if s.get("type") == "TextQuoteSelector":
                quote_sel = s

        if pos_sel is None or quote_sel is None:
            continue

        start = int(pos_sel["start"])
        end = int(pos_sel["end"])
        text = quote_sel["exact"]

        # --- Dedup (same offset + label) ---
        key = (start, end, label)
        if key in seen:
            continue
        seen.add(key)

        # --- Pre-fill IDs from Recogito comment body ---
        comment = get_comment_text(ann.get("body", []))
        wikidata_qid, geonames_id = extract_ids_from_comment(comment)
        note = ""
        if comment and not wikidata_qid and not geonames_id:
            # Non-link comment (e.g. "artwork", "forest") — useful context
            note = comment[:80]

        rows.append({
            "text": text,
            "label": label,
            "start_char": start,
            "end_char": end,
            "wikidata_qid": wikidata_qid or "",
            "geonames_id": geonames_id or "",
            "note": note,
        })

    # --- Write CSV ---
    df = pd.DataFrame(rows, columns=[
        "text", "label", "start_char", "end_char",
        "wikidata_qid", "geonames_id", "note",
    ])
    df.sort_values(["start_char", "end_char"], inplace=True)

    out_path = GS_FILE.with_name("gs_el_gold_template.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Written: {out_path}")
    print(f"  Entities: {len(df)}")
    print(f"  Labels:   {df['label'].value_counts().to_dict()}")
    print(f"  Pre-filled Wikidata: {df['wikidata_qid'].astype(bool).sum()}")
    print(f"  Pre-filled GeoNames: {df['geonames_id'].astype(bool).sum()}")
    print(f"  No link (blank):    {(~df['wikidata_qid'].astype(bool) & ~df['geonames_id'].astype(bool)).sum()}")
    print()
    print("Fill in remaining blank 'wikidata_qid' / 'geonames_id' columns manually.")
    print("Leave blank for entities that should not be linked (generic types, uncertain entities).")
    print("The 'note' column shows comments from the Recogito annotation for context.")


if __name__ == "__main__":
    main()
"""Map manually-entered Wikidata/GeoNames IDs from an older EL gold-standard CSV
onto a newly generated template CSV, avoiding manual re-entry.

The old CSV (`entity_linking/1816_el_gs.csv`) was manually completed based on an
earlier version of the ground-truth annotations. The new template
(`ner/gs_annotations/gs_el_gold_template.csv`) was generated from updated
annotations via `export_gs_to_el_template.py` and has mostly blank ID columns.

Matching is done on the composite key (text, label, start_char, end_char).
Values are transferred only when the new row's field is empty; pre-filled values
in the new template (from Recogito comment bodies) are preserved.  The old CSV's
"NIL" marker (meaning "reviewed, no link exists") is transferred as-is.

New entities that have no match in the old CSV are listed for manual follow-up.
"""

import pandas as pd
from pathlib import Path

OLD_CSV = Path("entity_linking/1816_el_gs.csv")
NEW_CSV = Path("ner/gs_annotations/gs_el_gold_template.csv")

KEY_COLS = ["text", "label", "start_char", "end_char"]
TRANSFER_COLS = ["wikidata_qid", "geonames_id", "note"]


def is_blank(val):
    """True if a cell is empty, NaN, or the string 'nan' (from CSV round-trip)."""
    if pd.isna(val):
        return True
    s = str(val).strip()
    return s == "" or s.lower() == "nan"


def normalize_id(val):
    """Normalize a Wikidata/GeoNames ID for comparison.

    Empty/NaN → None.
    Float-like integers (e.g. 2935704.0) → int string (e.g. '2935704').
    Everything else → stripped string as-is (including 'NIL').
    """
    if is_blank(val):
        return None
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val).strip()


def main():
    old = pd.read_csv(OLD_CSV)
    new = pd.read_csv(NEW_CSV)

    # Ensure ID columns are object dtype so we can store strings (old CSV has
    # strings like "NIL"; new CSV has float NaN for blanks → float64 dtype).
    for col in ["wikidata_qid", "geonames_id", "note"]:
        new[col] = new[col].astype(object)

    print(f"Old CSV: {len(old)} rows  ({OLD_CSV})")
    print(f"New CSV: {len(new)} rows  ({NEW_CSV})")

    # ── Left-join: new ← old ──────────────────────────────────────────────
    merged = new.merge(
        old[KEY_COLS + TRANSFER_COLS],
        on=KEY_COLS,
        how="left",
        suffixes=("_new", "_old"),
    )

    transferred_wd = 0
    transferred_gn = 0
    transferred_note = 0
    nil_wd = 0
    nil_gn = 0
    conflicts = 0

    for idx, row in merged.iterrows():
        for col in TRANSFER_COLS:
            old_val = row.get(f"{col}_old")
            new_val = row.get(f"{col}_new")

            old_norm = normalize_id(old_val)
            new_norm = normalize_id(new_val)

            # Nothing to transfer
            if old_norm is None:
                continue

            # Conflict: new already has a different non-blank value
            if new_norm is not None and new_norm != old_norm:
                conflicts += 1
                print(f"  ⚠ CONFLICT [{row['text']}] {col}: "
                      f"old='{old_norm}' vs new='{new_norm}' — keeping new")
                continue

            # Transfer: new is blank, or both are the same (no-op)
            if new_norm is None:
                merged.at[idx, f"{col}_new"] = old_val
                if col == "wikidata_qid":
                    transferred_wd += 1
                    if old_norm == "NIL":
                        nil_wd += 1
                elif col == "geonames_id":
                    transferred_gn += 1
                    if old_norm == "NIL":
                        nil_gn += 1
                elif col == "note":
                    transferred_note += 1

    # ── Build output DataFrame ────────────────────────────────────────────
    out = merged[KEY_COLS + [f"{c}_new" for c in TRANSFER_COLS]].copy()
    out.columns = KEY_COLS + TRANSFER_COLS
    out.sort_values(["start_char", "end_char"], inplace=True)

    # ── Write ──────────────────────────────────────────────────────────────
    out.to_csv(NEW_CSV, index=False, encoding="utf-8-sig")
    print(f"\nWritten: {NEW_CSV}")

    # ── Summary ────────────────────────────────────────────────────────────
    matched = merged["wikidata_qid_old"].notna().sum()
    unmatched = merged["wikidata_qid_old"].isna().sum()

    print(f"\n── Summary ──")
    print(f"  Matched (old → new):  {matched}")
    print(f"  New entities (blank): {unmatched}")
    print(f"  Wikidata QIDs transferred: {transferred_wd}  (NIL: {nil_wd})")
    print(f"  GeoNames IDs transferred:  {transferred_gn}  (NIL: {nil_gn})")
    print(f"  Notes transferred:         {transferred_note}")
    if conflicts:
        print(f"  Conflicts (kept new):      {conflicts}")

    still_blank_wd = out["wikidata_qid"].apply(is_blank).sum()
    still_blank_gn = out["geonames_id"].apply(is_blank).sum()
    print(f"  Still-blank Wikidata: {still_blank_wd}")
    print(f"  Still-blank GeoNames: {still_blank_gn}")

    # ── List new entities needing manual annotation ────────────────────────
    new_entities = merged[merged["wikidata_qid_old"].isna()]
    if len(new_entities) > 0:
        print(f"\n── New entities to annotate manually ({len(new_entities)}) ──")
        for _, r in new_entities.iterrows():
            pre = ""
            if not is_blank(r.get("wikidata_qid_new")):
                pre += f" [WD pre-filled: {r['wikidata_qid_new']}]"
            if not is_blank(r.get("geonames_id_new")):
                pre += f" [GN pre-filled: {r['geonames_id_new']}]"
            print(f"  [{r['label']}] \"{r['text']}\"  ({r['start_char']}–{r['end_char']}){pre}")


if __name__ == "__main__":
    main()

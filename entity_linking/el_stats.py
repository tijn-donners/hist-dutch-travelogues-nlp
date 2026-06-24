"""Simple script to show entity-linking statistics for an _el.spacy file.

Usage:
    python entity_linking/el_stats.py
"""

from pathlib import Path

import spacy
from spacy.tokens import Span, Token
from spacy.tokens import DocBin

# Register EL extension attributes (must happen before loading DocBin)
if not Span.has_extension("kb_id_wikidata_"):
    Span.set_extension("kb_id_wikidata_", default=None)
if not Span.has_extension("kb_id_geonames_"):
    Span.set_extension("kb_id_geonames_", default=None)

EL_RESULTS_DIR = Path(__file__).resolve().parent / "el-results"

# ── File selection ──────────────────────────────────────────────────────────

spacy_files = sorted(EL_RESULTS_DIR.glob("*_el.spacy"))
if not spacy_files:
    print(f"No _el.spacy files found in {EL_RESULTS_DIR}")
    raise SystemExit(1)

if len(spacy_files) == 1:
    spacy_path = spacy_files[0]
    print(f"Auto-selected: {spacy_path.name}")
else:
    print("Available _el.spacy files:")
    for i, f in enumerate(spacy_files, 1):
        print(f"  [{i}] {f.name}")
    choice = input("Select number: ").strip()
    try:
        idx = int(choice) - 1
        spacy_path = spacy_files[idx]
    except (ValueError, IndexError):
        print(f"Invalid selection")
        raise SystemExit(1)

# ── Load and analyze ───────────────────────────────────────────────────────

nlp = spacy.blank("nl")
db = DocBin().from_disk(str(spacy_path))
docs = list(db.get_docs(nlp.vocab))

total = 0
wikidata = 0
geonames = 0
both = 0
neither = 0

errors_wd = 0
errors_gn = 0

for doc in docs:
    for ent in doc.ents:
        total += 1
        wd = ent._.kb_id_wikidata_
        gn = ent._.kb_id_geonames_
        is_err_wd = isinstance(wd, str) and wd.startswith("ERROR:")
        is_err_gn = isinstance(gn, str) and gn.startswith("ERROR:")
        if is_err_wd:
            errors_wd += 1
        if is_err_gn:
            errors_gn += 1
        has_wd = wd is not None and not is_err_wd
        has_gn = gn is not None and not is_err_gn
        if has_wd:
            wikidata += 1
        if has_gn:
            geonames += 1
        if has_wd and has_gn:
            both += 1
        if not has_wd and not has_gn:
            neither += 1

print(f"\n{'='*50}")
print(f"File: {spacy_path.name}")
print(f"{'='*50}")
if total == 0:
    print("No entities found.")
else:
    print(f"Total entities:               {total}")
    print(f"  → Linked to Wikidata:       {wikidata:>4}  ({wikidata/total*100:5.1f}%)")
    print(f"  → Linked to GeoNames:        {geonames:>4}  ({geonames/total*100:5.1f}%)")
    print(f"  → Linked to both KBs:        {both:>4}  ({both/total*100:5.1f}%)")
    print(f"  → Not linked to either:      {neither:>4}  ({neither/total*100:5.1f}%)")
    if errors_wd or errors_gn:
        print()
        print(f"  ⚠ Errors:")
        print(f"     Wikidata rerank/select errors:  {errors_wd}")
        print(f"     GeoNames rerank/select errors:   {errors_gn}")
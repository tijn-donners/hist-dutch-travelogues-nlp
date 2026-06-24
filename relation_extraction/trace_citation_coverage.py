"""Trace how faithfully the RDF reflects the LLM's toponym_relations classification.

For each mention_id in ``toponym_relations`` (the LLM's explicit per-entity
classification), check whether it is actually linked to a travel event in the
generated .ttl — i.e. whether its ``LOC.*`` URI is the object of a VISIT
property (P7_took_place_at, P26_moved_to, P27_moved_from,
happened_in_proximity_of).

This separates two distinct causes of a JSON-vs-RDF evaluation gap:

* **LLM non-compliance** — a VISITED entity (IN/NEAR/THRU/TO/FROM) in
  ``toponym_relations`` that the LLM did NOT place in the corresponding
  event-role, so it has no VISIT triple in the RDF.
* **Deduplication / contamination** — a NO_REL mention whose ``LOC.*`` URI is
  shared (via the KB-ID cache) with a visited place, so it looks visited in
  the RDF even though the LLM classified it NO_REL.

The script changes nothing — it is purely diagnostic.

Usage:
    python relation_extraction/trace_citation_coverage.py [path/to/events.ttl]
"""

import json
import sys
from collections import Counter
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDF

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

RDF_DIR = ROOT_DIR / "output" / "rdf"
RE_DIR = ROOT_DIR / "output" / "re"

# Namespaces
CRM = "http://www.cidoc-crm.org/cidoc-crm/"
ATO = "http://academictourism.com/entity/"
ACADT = "http://academictourism.com/academictourism#"

E53 = URIRef(CRM + "E53_Place")
E18 = URIRef(CRM + "E18_Physical_Thing")
P129i = URIRef(CRM + "P129i_is_subject_of")

# Properties that indicate a location was visited / linked to a travel event
VISIT_PROPS = {
    URIRef(CRM + "P7_took_place_at"),
    URIRef(CRM + "P26_moved_to"),
    URIRef(CRM + "P27_moved_from"),
    URIRef(ACADT + "happened_in_proximity_of"),
}

VISITED_CATS = {"IN", "NEAR", "THRU", "TO", "FROM"}


# ---------------------------------------------------------------------------
# File selection (mirrors re_evaluate.py)
# ---------------------------------------------------------------------------
def select_ttl_file():
    """Scan output/rdf/ for *_events.ttl files and let the user pick one."""
    ttl_files = sorted(RDF_DIR.glob("*_events.ttl"))
    if not ttl_files:
        print(f"No *_events.ttl files found in {RDF_DIR}")
        print("Run relation_extraction/rel_extraction.py first.")
        raise SystemExit(1)

    if len(ttl_files) == 1:
        print(f"Auto-selected: {ttl_files[0].name}")
        return str(ttl_files[0])

    print("Available events TTL files:")
    for i, f in enumerate(ttl_files, 1):
        print(f"  [{i}] {f.name}")
    choice = input("Select number: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(ttl_files):
            return str(ttl_files[idx])
    except ValueError:
        pass
    print(f"Invalid selection: {choice}")
    raise SystemExit(1)


def find_matching_files(ttl_path):
    """Given a TTL path, find the corresponding events JSON and mention map."""
    stem = Path(ttl_path).stem
    base = stem[:-7] if stem.endswith("_events") else stem
    events_json = RE_DIR / f"{base}_events.json"
    mention_map_json = RE_DIR / f"{base}_mention_map.json"
    if not events_json.exists():
        print(f"ERROR: expected events JSON not found: {events_json}")
        raise SystemExit(1)
    if not mention_map_json.exists():
        print(f"ERROR: expected mention map not found: {mention_map_json}")
        raise SystemExit(1)
    return str(events_json), str(mention_map_json)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def derive_loc_visited(g):
    """Return (loc_entities, visited_locs) from the RDF graph.

    loc_entities : set of LOC.* URIs typed E53_Place or E18_Physical_Thing
    visited_locs : subset of loc_entities that are the object of a VISIT property
    """
    loc_entities = set()
    for s in g.subjects(RDF.type, E53):
        if str(s).startswith(ATO + "LOC."):
            loc_entities.add(s)
    for s in g.subjects(RDF.type, E18):
        if str(s).startswith(ATO + "LOC."):
            loc_entities.add(s)

    visited_locs = set()
    for loc in loc_entities:
        for s, p, o in g.triples((None, None, loc)):
            if p in VISIT_PROPS:
                visited_locs.add(loc)
                break
    return loc_entities, visited_locs


def map_mentions_to_loc(g, loc_entities, mention_map, letter_id):
    """Return dict mention_id → LOC.* URI for mentions whose CT points to a
    LOC.* entity via P129i_is_subject_of."""
    ct_prefix = ATO + f"CT.{letter_id}."
    mid_to_loc = {}
    for loc in loc_entities:
        for ct_entity in g.objects(loc, P129i):
            ct_str = str(ct_entity)
            if ct_str.startswith(ct_prefix):
                mid = ct_str[len(ct_prefix):]
                if mid in mention_map and mention_map[mid]["label"] in (
                    "E53_Place", "E18_Physical_Thing"):
                    mid_to_loc[mid] = loc
    return mid_to_loc


def report(ttl_path, events_json, mention_map_json):
    with open(events_json, encoding="utf-8") as f:
        events_data = json.load(f)
    with open(mention_map_json, encoding="utf-8") as f:
        mention_map_raw = json.load(f)
    mention_map = {
        mid: {"label": info.get("label", ""), "text": info.get("text", "")}
        for mid, info in mention_map_raw.items()
    }

    letter_id = events_data.get("letter_id", "BRF0003")
    toponym_relations = events_data.get("toponym_relations", {})

    g = Graph()
    g.parse(ttl_path, format="turtle")

    loc_entities, visited_locs = derive_loc_visited(g)
    mid_to_loc = map_mentions_to_loc(g, loc_entities, mention_map, letter_id)

    # Per-category breakdown
    cats = ["IN", "NEAR", "THRU", "TO", "FROM", "NO_REL"]
    rows = []
    total_visited = 0
    total_visited_linked = 0
    total_noncompliance = 0
    total_contamination = 0

    for cat in cats:
        mids = [m for m, c in toponym_relations.items() if c == cat]
        total = len(mids)
        linked = 0
        not_linked_mids = []
        for mid in mids:
            loc = mid_to_loc.get(mid)
            if loc is not None and loc in visited_locs:
                linked += 1
            else:
                not_linked_mids.append(mid)
        rows.append((cat, total, linked, len(not_linked_mids), not_linked_mids))
        if cat in VISITED_CATS:
            total_visited += total
            total_visited_linked += linked
            total_noncompliance += len(not_linked_mids)
        else:  # NO_REL
            total_contamination += linked

    # --- Print report ---
    print("=" * 78)
    print("Toponym-relations (JSON output) → RDF citation coverage (.ttl)")
    print(f"  TTL:      {Path(ttl_path).name}")
    print(f"  letter_id: {letter_id}")
    print("=" * 78)
    print()
    print(f"{'category':<10} {'total':>6} {'linked':>8} "
          f"{'not-linked':>12}   note")
    print(f"{'':<10} {'':>6} {'':>8} {'':>12}")
    print("-" * 78)
    for cat, total, linked, n_not, _ in rows:
        note = ""
        if cat == "NO_REL" and linked > 0:
            note = f"<- {linked} contaminated (shared LOC.* is visited)"
        elif cat in VISITED_CATS and n_not > 0:
            note = "<- LLM non-compliance (not in any event-role)"
        print(f"{cat:<10} {total:>6} {linked:>8} {n_not:>12}   {note}")
    print("-" * 78)

    coverage = (total_visited_linked / total_visited
                if total_visited else 0.0)
    print()
    print(f"Coverage (visited & linked / total visited): "
          f"{total_visited_linked}/{total_visited} ({coverage:.1%})")
    print(f"LLM non-compliance (visited & NOT linked):  {total_noncompliance}")
    print(f"Contamination (NO_REL but linked via shared LOC): {total_contamination}")
    print()

    # --- Detail: which VISITED entities are not linked (non-compliance) ---
    if total_noncompliance:
        print("=" * 78)
        print(f"Detail: VISITED entities NOT linked to any travel event "
              f"(LLM non-compliance, {total_noncompliance})")
        print("=" * 78)
        for cat, total, linked, n_not, not_linked_mids in rows:
            if cat not in VISITED_CATS or not not_linked_mids:
                continue
            print(f"\n  {cat} ({n_not}):")
            for mid in sorted(not_linked_mids, key=lambda x: int(x[1:])
                               if x[1:].isdigit() else 0):
                info = mention_map.get(mid, {})
                print(f"    {mid}: {info.get('text', '')[:50]!r}")
        print()

    # --- Detail: contaminated NO_REL ---
    if total_contamination:
        print("=" * 78)
        print(f"Detail: NO_REL entities that LOOK visited in RDF "
              f"(contamination, {total_contamination})")
        print("=" * 78)
        for cat, total, linked, n_not, not_linked_mids in rows:
            if cat != "NO_REL":
                continue
            contam = [m for m in toponym_relations if toponym_relations[m] == "NO_REL"
                      and mid_to_loc.get(m) in visited_locs]
            for mid in sorted(contam, key=lambda x: int(x[1:])
                              if x[1:].isdigit() else 0):
                info = mention_map.get(mid, {})
                loc = mid_to_loc.get(mid)
                label = g.value(loc, URIRef("http://www.w3.org/2000/01/rdf-schema#label"))
                print(f"  {mid}: {info.get('text', '')[:30]!r}  "
                      f"-> LOC.* = {label}")
            print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    path = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            path = arg
            break

    if path is None:
        path = select_ttl_file()
    elif not Path(path).exists():
        print(f"File not found: {path}")
        sys.exit(1)

    events_json, mention_map_json = find_matching_files(path)
    report(path, events_json, mention_map_json)
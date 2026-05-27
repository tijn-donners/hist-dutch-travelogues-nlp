"""Validate generated RDF against ATO schema constraints.

Usage:
    python relation_extraction/validate_rdf.py [path/to/events.ttl]
"""

import sys
from pathlib import Path

from rdflib import RDF, RDFS, Graph
from rdflib.term import URIRef

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

CIDOC = "http://www.cidoc-crm.org/cidoc-crm/"
ATO = "http://academictourism.com/entity/"

E7 = URIRef(CIDOC + "E7_Activity")
E9 = URIRef(CIDOC + "E9_Move")
E53 = URIRef(CIDOC + "E53_Place")
E18 = URIRef(CIDOC + "E18_Physical_Thing")
E74 = URIRef(CIDOC + "E74_Group")
F2 = URIRef("http://iflastandards.info/ns/lrm/lrmoo/F2_Expression")

P183 = URIRef(CIDOC + "P183_ends_before_the_start_of")
P10 = URIRef(CIDOC + "P10_contains")
P10f = URIRef(CIDOC + "P10_falls_within")
P26 = URIRef(CIDOC + "P26_moved_to")
P27 = URIRef(CIDOC + "P27_moved_from")
P7 = URIRef(CIDOC + "P7_took_place_at")
P8 = URIRef(CIDOC + "P8_took_place_on_or_within")
P67 = URIRef(CIDOC + "P67_refers_to")
P11 = URIRef(CIDOC + "P11_had_participant")


def validate(ttl_path):
    g = Graph()
    g.parse(ttl_path, format="turtle")
    print(f"Loaded {len(g)} triples from {ttl_path}\n")

    errors = []
    warnings = []

    # --- 1. Every event must be typed E7_Activity or E9_Move ---
    events = set(g.subjects(RDF.type, E7)) | set(g.subjects(RDF.type, E9))
    # Also find events via P10_falls_within (sub-events)
    for s in g.subjects(P10f, None):
        events.add(s)
    # Also find events via P183 (should all be events)
    for s in g.subjects(P183, None):
        events.add(s)
    for s in g.objects(None, P183):
        events.add(s)
    # Filter out non-event URIs (like the letter itself, places)
    events = {e for e in events if "RS" in str(e) or "BRF" in str(e)}

    untyped = [e for e in events if (e, RDF.type, E7) not in g and (e, RDF.type, E9) not in g]
    if untyped:
        errors.append(f"{len(untyped)} events not typed as E7_Activity or E9_Move: {untyped}")

    # --- 2. Every E9_Move (except the overall journey container) must have P26 and P27 ---
    moves = set(g.subjects(RDF.type, E9))
    for m in moves:
        # The root .RS node is a mereological container, not a real translocation
        if _is_journey_root(m):
            continue
        if (m, P26, None) not in g:
            errors.append(f"E9_Move {_short(m)} missing P26_moved_to")
        if (m, P27, None) not in g:
            errors.append(f"E9_Move {_short(m)} missing P27_moved_from")

    # --- 3. Every E7_Activity (non-move) should have P7 (location) ---
    activities = set(g.subjects(RDF.type, E7)) - moves
    for a in activities:
        if (a, P7, None) not in g:
            warnings.append(f"E7_Activity {_short(a)} has no P7_took_place_at")

    # --- 4. Every place referenced in P7/P26/P27 must be typed E53_Place ---
    for pred in [P7, P26, P27]:
        for s, o in g.subject_objects(pred):
            if (o, RDF.type, E53) not in g:
                errors.append(f"Place {_short(o)} (referenced by {_short(s)}) not typed as E53_Place")

    # --- 5. Every P8 target must be E18_Physical_Thing ---
    for s, o in g.subject_objects(P8):
        if (o, RDF.type, E18) not in g:
            warnings.append(f"P8 target {_short(o)} (from {_short(s)}) not typed as E18_Physical_Thing")

    # --- 6. P183 chain must be unbroken ---
    p183_targets = set(g.objects(None, P183))
    p183_sources = set(g.subjects(P183, None))
    chain_events = p183_sources | p183_targets
    if chain_events:
        first_events = p183_sources - p183_targets
        last_events = p183_targets - p183_sources
        if len(first_events) != 1:
            errors.append(f"Expected 1 first event (no predecessor), found {len(first_events)}: {[_short(e) for e in first_events]}")
        if len(last_events) != 1:
            errors.append(f"Expected 1 last event (no successor), found {len(last_events)}: {[_short(e) for e in last_events]}")
        # Check for events not in the chain at all
        missing_from_chain = events - chain_events - {e for e in events if str(e).endswith(".RS")}
        if missing_from_chain:
            warnings.append(f"{len(missing_from_chain)} events not in P183 chain: {[_short(e) for e in missing_from_chain]}")

    # --- 7. No duplicate event IDs ---
    seen = {}
    for e in events:
        key = str(e)
        if key in seen:
            errors.append(f"Duplicate event ID: {key}")
        seen[key] = True

    # --- 8. P67 targets should not also be visited (P7/P26/P27) ---
    p67_targets = set(g.objects(None, P67))
    visited_targets = set(g.objects(None, P7)) | set(g.objects(None, P26)) | set(g.objects(None, P27))
    confused = p67_targets & visited_targets
    if confused:
        errors.append(f"Places both mentioned (P67) and visited (P7/P26/P27): {[_short(p) for p in confused]}")

    # --- 9. Overall journey must have P10_contains ---
    seen_journeys = set()
    for s in g.subjects(P10, None):
        if s in seen_journeys:
            continue
        seen_journeys.add(s)
        contained = list(g.objects(s, P10))
        if contained:
            label = g.value(s, RDFS.label)
            print(f"  Journey {_short(s)} ('{label}') contains {len(contained)} events")

    # --- 10. Check for required participant/protagonist links ---
    for e in events:
        if (e, P11, None) not in g:
            warnings.append(f"Event {_short(e)} has no P11_had_participant")

    # --- 11. Event ID convention: check for BRF prefix and RS pattern ---
    for e in events:
        s = str(e)
        if not s.startswith(ATO + "BRF"):
            warnings.append(f"Event {_short(e)} does not start with BRF prefix")
        if "RS" not in s:
            warnings.append(f"Event {_short(e)} missing RS (Reis) in ID")

    # --- Report ---
    print("=" * 60)
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for err in errors:
            print(f"  [ERROR] {err}")
    else:
        print("No errors found.")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN]  {w}")
    else:
        print("No warnings.")

    print(f"\nSummary: {len(errors)} errors, {len(warnings)} warnings, {len(events)} events checked")

    return len(errors) == 0


def _short(uri):
    """Return a compact form of a URI."""
    s = str(uri)
    if s.startswith(ATO):
        return s[len(ATO):]
    return s


def _is_journey_root(uri):
    """Check if a URI is a root journey node like BRF0003.RS (not a real event)."""
    s = str(uri)
    return s.count(".RS") == 1 and s.endswith(".RS")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT_DIR / "output" / "rdf" / "1816_third_letter_events.ttl")
    if not Path(path).exists():
        print(f"File not found: {path}")
        sys.exit(1)
    ok = validate(path)
    sys.exit(0 if ok else 1)

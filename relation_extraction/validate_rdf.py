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
LRM = "http://iflastandards.info/ns/lrm/lrmoo/"
ACADEMICTOURISM = "http://academictourism.com/academictourism#"

# CIDOC CRM types
E1 = URIRef(CIDOC + "E1_CRM_Entity")
E2 = URIRef(CIDOC + "E2_Temporal_Entity")
E4 = URIRef(CIDOC + "E4_Period")
E5 = URIRef(CIDOC + "E5_Event")
E7 = URIRef(CIDOC + "E7_Activity")
E9 = URIRef(CIDOC + "E9_Move")
E18 = URIRef(CIDOC + "E18_Physical_Thing")
E19 = URIRef(CIDOC + "E19_Physical_Object")
E20 = URIRef(CIDOC + "E20_Biological_Object")
E22 = URIRef(CIDOC + "E22_Human-Made_Object")
E52 = URIRef(CIDOC + "E52_Time-Span")
E53 = URIRef(CIDOC + "E53_Place")
E55 = URIRef(CIDOC + "E55_Type")
E74 = URIRef(CIDOC + "E74_Group")
E89 = URIRef(CIDOC + "E89_Propositional_Object")
E92 = URIRef(CIDOC + "E92_Spacetime_Volume")
F2 = URIRef(LRM + "F2_Expression")

# Properties
P4 = URIRef(CIDOC + "P4_has_time-span")
P7 = URIRef(CIDOC + "P7_took_place_at")
P8 = URIRef(CIDOC + "P8_took_place_on_or_within")
P10 = URIRef(CIDOC + "P10_contains")
P10f = URIRef(CIDOC + "P10_falls_within")
P11 = URIRef(CIDOC + "P11_had_participant")
P12 = URIRef(CIDOC + "P12_occurred_in_the_presence_of")
P16 = URIRef(CIDOC + "P16_used_specific_object")
P26 = URIRef(CIDOC + "P26_moved_to")
P27 = URIRef(CIDOC + "P27_moved_from")
P59 = URIRef(CIDOC + "P59_has_section")
P67 = URIRef(CIDOC + "P67_refers_to")
P67i = URIRef(CIDOC + "P67i_is_referred_to_by")
P101 = URIRef(CIDOC + "P101_had_as_general_use")
P129 = URIRef(CIDOC + "P129_is_about")
P182 = URIRef(CIDOC + "P182_ends_before_or_with_the_start_of")
P183 = URIRef(CIDOC + "P183_ends_before_the_start_of")
CONVEYS = URIRef(ACADEMICTOURISM + "conveys")


def validate(ttl_path):
    g = Graph()
    g.parse(ttl_path, format="turtle")
    print(f"Loaded {len(g)} triples from {ttl_path}\n")

    errors = []
    warnings = []

    # --- 1. Every event must have full CIDOC CRM type hierarchy ---
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

    # Check full type hierarchy on each event
    for e in events:
        for required_type in [E1, E2, E4, E5, E7, E92]:
            if (e, RDF.type, required_type) not in g:
                warnings.append(f"Event {_short(e)} missing type {_short(required_type)}")

    untyped = [e for e in events if (e, RDF.type, E7) not in g and (e, RDF.type, E9) not in g]
    if untyped:
        errors.append(f"{len(untyped)} events not typed as E7_Activity or E9_Move: {untyped}")

    # --- 2. Every E9_Move (except the overall journey container) must have P26 and P27 ---
    moves = set(g.subjects(RDF.type, E9))
    for m in moves:
        if _is_journey_root(m):
            continue
        if (m, P26, None) not in g:
            errors.append(f"E9_Move {_short(m)} missing P26_moved_to")
        if (m, P27, None) not in g:
            errors.append(f"E9_Move {_short(m)} missing P27_moved_from")

    # --- 3. Every E7_Activity (non-move, non-joining, non-leaving) should have P7 ---
    activities = set(g.subjects(RDF.type, E7)) - moves
    for a in activities:
        if (a, P7, None) not in g and (a, P8, None) not in g:
            warnings.append(f"E7_Activity {_short(a)} has no P7_took_place_at or P8_took_place_on_or_within")

    # --- 4. Every place referenced in P7/P26/P27 must be typed E53_Place ---
    for pred in [P7, P26, P27]:
        for s, o in g.subject_objects(pred):
            if (o, RDF.type, E53) not in g:
                errors.append(f"Place {_short(o)} (referenced by {_short(s)}) not typed as E53_Place")

    # --- 5. Every P8 target must be E18_Physical_Thing or subclass (E19/E20/E22) ---
    for s, o in g.subject_objects(P8):
        if (o, RDF.type, E18) not in g and (o, RDF.type, E19) not in g and (o, RDF.type, E20) not in g and (o, RDF.type, E22) not in g:
            warnings.append(f"P8 target {_short(o)} (from {_short(s)}) not typed as E18_Physical_Thing or subclass")

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

    # --- 8. Letter-level P67 targets should not also be visited (P7/P26/P27) ---
    # NB: per-mention CT.* citations are also typed F2_Expression and carry their
    # own P67_refers_to to the entity (per ATO_paper §5), so a global
    # g.objects(None, P67) would include every visited place via its CT. We scope
    # this rule to the *letter's* P67 objects only (the "mentioned-only" set),
    # matching query_route's mentioned_toponyms. The letter is the F2_Expression
    # subject whose URI has no "."-segment after the ato: prefix (CT.*/ART.* do).
    letter_uris = {s for s in g.subjects(RDF.type, F2)
                   if str(s).startswith(ATO) and "." not in str(s)[len(ATO):]}
    letter_p67 = set()
    for lu in letter_uris:
        letter_p67 |= set(g.objects(lu, P67))
    visited_targets = set(g.objects(None, P7)) | set(g.objects(None, P26)) | set(g.objects(None, P27))
    confused = letter_p67 & visited_targets
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

    # --- 12. Every event with time_span citations should have P4_has_time-span ---
    for e in events:
        has_time_citation = False
        for ct_uri in g.objects(e, CONVEYS):
            ct_label = str(g.value(ct_uri, RDFS.label) or "")
            if "Time_Span" in ct_label:
                has_time_citation = True
                break
        if has_time_citation and (e, P4, None) not in g:
            warnings.append(f"Event {_short(e)} has Time_Span citations but no P4_has_time-span")

    # --- 13. Every P4_has_time-span target must be typed E52_Time-Span ---
    for s, o in g.subject_objects(P4):
        if (o, RDF.type, E52) not in g:
            errors.append(f"P4 target {_short(o)} (from {_short(s)}) not typed as E52_Time-Span")

    # --- 14. P59_has_section domain=E18, range=E53 ---
    for s, o in g.subject_objects(P59):
        if (s, RDF.type, E18) not in g:
            errors.append(f"P59_has_section domain {_short(s)} not typed as E18_Physical_Thing")
        if (o, RDF.type, E53) not in g:
            errors.append(f"P59_has_section range {_short(o)} not typed as E53_Place")

    # --- 15. P101_had_as_general_use range=E55_Type ---
    for s, o in g.subject_objects(P101):
        if (o, RDF.type, E55) not in g:
            warnings.append(f"P101 target {_short(o)} (from {_short(s)}) not typed as E55_Type")

    # --- 16. P16_used_specific_object range=E22_Human-made_Object ---
    for s, o in g.subject_objects(P16):
        if (o, RDF.type, E22) not in g:
            warnings.append(f"P16 target {_short(o)} (from {_short(s)}) not typed as E22_Human-Made_Object")

    # --- 17. Letter (F2_Expression) should have P129_is_about → journey ---
    # NB: per-mention CT.* citations and artworks are also F2_Expression but have
    # no P129_is_about; restrict the focus set to the letter (F2 subject whose
    # URI has no "."-segment after the ato: prefix).
    letters = {s for s in g.subjects(RDF.type, F2)
               if str(s).startswith(ATO) and "." not in str(s)[len(ATO):]}
    for letter in letters:
        if (letter, P129, None) not in g:
            warnings.append(f"Letter {_short(letter)} missing P129_is_about (journey subject)")

    # --- 18. P12_occurred_in_the_presence_of targets should be typed ---
    for s, o in g.subject_objects(P12):
        # Skip Tour Group (E74) — that's always present
        if (o, RDF.type, E74) in g:
            continue
        # Check that the target has at least one meaningful type
        has_type = any((o, RDF.type, t) in g for t in [E18, E19, E20, E22, E53, F2])
        if not has_type:
            warnings.append(f"P12 target {_short(o)} (from {_short(s)}) has no recognized type")

    # --- 19. VP* sub-moves (THRU via-points) must have P26 and P27 ---
    vp_moves = {e for e in events if ".VP" in str(e)}
    for vp in sorted(vp_moves, key=str):
        if (vp, P26, None) not in g:
            errors.append(f"THRU sub-move {_short(vp)} missing P26_moved_to")
        if (vp, P27, None) not in g:
            errors.append(f"THRU sub-move {_short(vp)} missing P27_moved_from")
        # Must have P9_falls_within to a parent translocation
        parent = g.value(vp, P10f)
        if parent is None:
            errors.append(f"THRU sub-move {_short(vp)} missing P9_falls_within "
                          f"(parent translocation)")
        elif (parent, RDF.type, E9) not in g:
            errors.append(f"THRU sub-move {_short(vp)} P9_falls_within target "
                          f"{_short(parent)} is not typed as E9_Move")

    # --- 20. happened_in_proximity_of targets must be typed E53_Place or E18_Physical_Thing ---
    PROXIMITY = URIRef(ACADEMICTOURISM + "happened_in_proximity_of")
    for s, o in g.subject_objects(PROXIMITY):
        if (o, RDF.type, E53) not in g and (o, RDF.type, E18) not in g:
            errors.append(f"happened_in_proximity_of target {_short(o)} "
                          f"(from {_short(s)}) not typed as E53_Place or "
                          f"E18_Physical_Thing")
        if (s, RDF.type, E5) not in g:
            errors.append(f"happened_in_proximity_of source {_short(s)} "
                          f"not typed as E5_Event")

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

    n_vp = len([e for e in events if ".VP" in str(e)])
    n_proximity = len(list(g.subject_objects(PROXIMITY)))
    print(f"\nSummary: {len(errors)} errors, {len(warnings)} warnings, "
          f"{len(events)} events checked ({n_vp} THRU sub-moves, "
          f"{n_proximity} happened_in_proximity_of)")

    return len(errors) == 0


def _short(uri):
    """Return a compact form of a URI."""
    s = str(uri)
    if s.startswith(ATO):
        return s[len(ATO):]
    if s.startswith(CIDOC):
        return s[len(CIDOC):]
    if s.startswith(LRM):
        return s[len(LRM):]
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

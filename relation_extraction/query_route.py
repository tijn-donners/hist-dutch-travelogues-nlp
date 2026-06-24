"""Query and display travel route from ATO RDF with full provenance.

Shows the chronological chain of events, nested sub-events, CT.* citation
mentions, mode of transport, mentioned-only places, and summary statistics.

Usage:
    python relation_extraction/query_route.py [path/to/events.ttl] [--csv]
"""

import sys
from pathlib import Path

from rdflib import RDF, RDFS, Graph
from rdflib.term import URIRef

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------
CIDOC_NS = "http://www.cidoc-crm.org/cidoc-crm/"
ATO_NS = "http://academictourism.com/entity/"
ACADT_NS = "http://academictourism.com/academictourism#"
LRMOO_NS = "http://iflastandards.info/ns/lrm/lrmoo/"
WD_NS = "https://www.wikidata.org/wiki/"

# CIDOC CRM properties
P183 = URIRef(CIDOC_NS + "P183_ends_before_the_start_of")
P26 = URIRef(CIDOC_NS + "P26_moved_to")
P27 = URIRef(CIDOC_NS + "P27_moved_from")
P7 = URIRef(CIDOC_NS + "P7_took_place_at")
P8 = URIRef(CIDOC_NS + "P8_took_place_on_or_within")
P10_contains = URIRef(CIDOC_NS + "P10_contains")
P10_falls = URIRef(CIDOC_NS + "P10_falls_within")
P67_refers = URIRef(CIDOC_NS + "P67_refers_to")
P129 = URIRef(CIDOC_NS + "P129_is_about")
E9_Move = URIRef(CIDOC_NS + "E9_Move")
E89 = URIRef(CIDOC_NS + "E89_Propositional_Object")

# Toponym (visitable location) types. A visitable toponym is an E53_Place, or
# an E18_Physical_Thing that is NOT also typed with a subclass of E18.
# Subclasses of E18 used in rel_extraction.py: vehicles (E22), museum objects
# (E19), biological specimens (E20) — none are places the traveler "visits".
E53 = URIRef(CIDOC_NS + "E53_Place")
E18 = URIRef(CIDOC_NS + "E18_Physical_Thing")
E18_SUBCLASSES = [
    URIRef(CIDOC_NS + "E19_Physical_Object"),
    URIRef(CIDOC_NS + "E20_Biological_Object"),
    URIRef(CIDOC_NS + "E22_Human_Made_Object"),
]

# SKOS
SKOS_NS = "http://www.w3.org/2004/02/skos/core#"
SKOS_closeMatch = URIRef(SKOS_NS + "closeMatch")

# ATO custom properties
CONVEYS = URIRef(ACADT_NS + "conveys")
PROXIMITY = URIRef(ACADT_NS + "happened_in_proximity_of")

# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------
def select_ttl_file():
    """Scan output/rdf/ for *_events.ttl files and let the user pick one."""
    rdf_dir = ROOT_DIR / "output" / "rdf"
    ttl_files = sorted(rdf_dir.glob("*_events.ttl"))

    if not ttl_files:
        print(f"No *_events.ttl files found in {rdf_dir}")
        raise SystemExit(1)

    if len(ttl_files) == 1:
        print(f"Auto-selected: {ttl_files[0].name}")
        return str(ttl_files[0])

    print("Available RDF files:")
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


# ---------------------------------------------------------------------------
# Route display
# ---------------------------------------------------------------------------
def query_route(ttl_path, csv_mode=False):
    """Load RDF and display the full travel route with provenance.

    Args:
        ttl_path: Path to the *_events.ttl Turtle file.
        csv_mode: If True, output CSV instead of human-readable table.
    """
    g = Graph()
    g.parse(ttl_path, format="turtle")

    # --- Resolve Wikidata prefix for display ---
    def wd(uri):
        """Shorten a Wikidata URI to wd:Qxxx, or return the full URI."""
        s = str(uri)
        return "wd:" + s.replace(WD_NS, "") if s.startswith(WD_NS) else s

    def short(event_uri):
        """Shorten an ATO event URI to its local ID."""
        return str(event_uri).replace(ATO_NS, "")

    def place_repr(place_uri):
        """Render a place with its Wikidata ID if known."""
        if place_uri is None:
            return ""
        label = g.value(place_uri, RDFS.label)
        qid = None
        for o in g.objects(place_uri, URIRef("http://www.w3.org/2004/02/skos/core#closeMatch")):
            qid = wd(o)
        label_str = str(label) if label else short(place_uri)
        return f"{label_str} ({qid})" if qid else label_str

    def is_toponym(place_uri):
        """A visitable toponym: E53_Place, or E18_Physical_Thing NOT also
        typed with an E18 subclass (E19/E20/E22).

        Excludes vehicles, museum objects, biological specimens, time-spans,
        artworks, documents, and transport modes — none are places the
        traveler visits. A building (E18+E53) is matched by the E53 branch;
        a room (E53 only) by the E53 branch; a vehicle (E18+E22, no E53) is
        rejected because E22 is an E18 subclass.
        """
        if place_uri is None:
            return False
        if (place_uri, RDF.type, E53) in g:
            return True
        if (place_uri, RDF.type, E18) in g and not any(
            (place_uri, RDF.type, sub) in g for sub in E18_SUBCLASSES
        ):
            return True
        return False

    # --- Build P183 chain ---
    pairs = list(g.subject_objects(P183))
    sources = {a for a, _ in pairs}
    targets = {b for _, b in pairs}
    first_events = sources - targets

    if not first_events:
        print("ERROR: no first event found (P183 chain has cycle or is empty)")
        return

    if len(first_events) > 1:
        print(f"WARNING: {len(first_events)} first events found, picking one")

    chain = []
    current = sorted(first_events)[0]
    while current:
        chain.append(current)
        next_events = [b for a, b in pairs if a == current]
        current = next_events[0] if len(next_events) == 1 else None

    # --- Build parent-child map (P10_contains) ---
    parent_of = {}  # parent -> [child, ...]
    child_parent = {}  # child -> parent
    for p, c in g.subject_objects(P10_contains):
        parent_of.setdefault(p, []).append(c)
        child_parent[c] = p

    # Find the journey root (the parent that isn't in the chain)
    journey_root = None
    for p in parent_of:
        if p not in chain and p not in child_parent:
            journey_root = p
            break

    # --- Collect cited events ---
    cited_events = set(g.subjects(CONVEYS))

    # --- Group sub-events using P10_contains from parent_of map ---
    sub_events_of = {}  # parent -> [child, ...]
    skip_in_chain = set()

    for ev in chain:
        children = parent_of.get(ev, [])
        if children:
            sub_events_of[ev] = sorted(children, key=str)
            skip_in_chain.update(children)

    # Sort sub-events by their position in the chain
    chain_index = {ev: i for i, ev in enumerate(chain)}
    for ev in sub_events_of:
        sub_events_of[ev].sort(key=lambda x: chain_index.get(x, 0))

    # --- Visited toponyms + roles (single shared pass) ---
    # Collected from chain events (non-skipped) and sub-events via the
    # location-typed properties P27/P26/P7/P8, filtered to visitable toponyms
    # only (E53_Place or pure E18_Physical_Thing — see is_toponym()).
    letter_uri = URIRef(ATO_NS + "BRF0003")
    visited_toponyms = set()  # uri -> seen
    place_roles = {}          # uri -> set of role strings
    place_event_counts = {}   # uri -> count of events referencing it

    def _record_place(uri, role):
        if uri is None or not is_toponym(uri):
            return
        visited_toponyms.add(uri)
        place_roles.setdefault(uri, set()).add(role)
        place_event_counts[uri] = place_event_counts.get(uri, 0) + 1

    for ev in chain:
        if ev in skip_in_chain:
            continue
        is_move = (ev, RDF.type, E9_Move) in g
        mfrom = g.value(ev, P27)
        mto = g.value(ev, P26)
        at_p = g.value(ev, P7)
        within = g.value(ev, P8)
        if is_move:
            _record_place(mfrom, "departure")
            _record_place(mto, "arrival")
        else:
            _record_place(at_p, "visited")
            _record_place(within, "visited (indoor)")
        # Sub-event locations (including VP* THRU sub-moves with P26/P27)
        for sub in sub_events_of.get(ev, []):
            sub_is_move = (sub, RDF.type, E9_Move) in g
            if sub_is_move:
                _record_place(g.value(sub, P27), "departure")
                _record_place(g.value(sub, P26), "arrival")
            else:
                _record_place(g.value(sub, P7), "visited")
                _record_place(g.value(sub, P8), "visited (indoor)")
        # Near places (happened_in_proximity_of)
        for near_place in g.objects(ev, PROXIMITY):
            _record_place(near_place, "near")
        for sub in sub_events_of.get(ev, []):
            for near_place in g.objects(sub, PROXIMITY):
                _record_place(near_place, "near")

    # --- Letter-level references (P67_refers_to) ---
    # Mentioned-only toponyms: things the letter refers to that are visitable
    # toponyms but were NOT visited (not in visited_toponyms). Non-toponym
    # P67_refers_to targets (time-spans, transport modes, artworks, documents,
    # vehicles, museum objects, specimens) are excluded by is_toponym().
    mentioned_toponyms = {
        mp for mp in g.objects(letter_uri, P67_refers)
        if is_toponym(mp) and mp not in visited_toponyms
    }

    # --- Citations for an event ---
    def get_citations(event_uri):
        cits = []
        for ct in sorted(g.objects(event_uri, CONVEYS), key=str):
            ct_label = str(g.value(ct, RDFS.label) or short(ct))
            qid = None
            for q in g.objects(ct, P129):
                match = g.value(q, SKOS_closeMatch)
                qid = wd(match) if match else None
            mid = short(ct).split(".")[-1]  # Extract eN from CT.BRF0003.eN
            cits.append((mid, ct_label, qid))
        return cits

    # --- Mode of transport for an event ---
    def get_mode(event_uri):
        """Return the mode-of-transport label for an event, or empty string."""
        for o in g.objects(event_uri, URIRef(CIDOC_NS + "P101_had_as_general_use")):
            mode_label = g.value(o, RDFS.label)
            if mode_label:
                return str(mode_label)
        return ""

    # --- Time spans for an event ---
    def get_time_spans(event_uri):
        """Return sorted list of time-span labels for an event."""
        spans = []
        for o in g.objects(event_uri, URIRef(CIDOC_NS + "P4_has_time-span")):
            ts_label = g.value(o, RDFS.label)
            if ts_label:
                spans.append(str(ts_label))
        return spans

    # --- CSV mode ---
    if csv_mode:
        print("step,event_id,parent_id,type,label,from,from_qid,to,to_qid,at,at_qid,within,within_qid,near,near_qid,mode,citations")

        def pq(uri):
            """Return (label, qid) for a place URI."""
            if uri is None:
                return ("", "")
            lbl = str(g.value(uri, RDFS.label) or "")
            for q in g.objects(uri, URIRef("http://www.w3.org/2004/02/skos/core#closeMatch")):
                return (lbl, wd(q))
            return (lbl, "")

        step = 0
        for ev in chain:
            if ev in skip_in_chain:
                continue
            step += 1
            ev_id = short(ev)
            label = str(g.value(ev, RDFS.label) or "")
            is_move = (ev, RDF.type, E9_Move) in g

            mfrom = g.value(ev, P27)
            mto = g.value(ev, P26)
            at = g.value(ev, P7)
            within = g.value(ev, P8)

            from_label, from_qid = pq(mfrom)
            to_label, to_qid = pq(mto)
            at_label, at_qid = pq(at)
            within_label, within_qid = pq(within)

            # Near places
            near_places = list(g.objects(ev, PROXIMITY))
            near_label = "; ".join(str(g.value(np, RDFS.label) or short(np)) for np in near_places) if near_places else ""
            near_qid = ""
            if near_places:
                qids = []
                for np in near_places:
                    for q in g.objects(np, URIRef("http://www.w3.org/2004/02/skos/core#closeMatch")):
                        qids.append(wd(q))
                near_qid = "; ".join(qids)

            if is_move:
                etype = "translocation"
            elif within:
                etype = "indoor_tour"
            else:
                etype = "outdoor_tour/stay"

            mode = ""
            if is_move and "met " in label.lower():
                mode = label.split(" met ")[-1].rstrip(".")

            cits = "; ".join(mid for mid, _, _ in get_citations(ev))
            parent_id = short(child_parent[ev]) if ev in child_parent else ""

            print(f'{step},"{ev_id}","{parent_id}","{etype}","{label}","{from_label}","{from_qid}","{to_label}","{to_qid}","{at_label}","{at_qid}","{within_label}","{within_qid}","{near_label}","{near_qid}","{mode}","{cits}"')

            # Sub-events
            for sub in sub_events_of.get(ev, []):
                sub_id = short(sub)
                sub_label = str(g.value(sub, RDFS.label) or "")
                sub_at = g.value(sub, P7)
                sub_within = g.value(sub, P8)
                at_l, at_q = pq(sub_at)
                w_l, w_q = pq(sub_within)
                sub_type = "sub_indoor" if sub_within else "sub_outdoor/stay"
                sub_cits = "; ".join(mid for mid, _, _ in get_citations(sub))
                sub_near = list(g.objects(sub, PROXIMITY))
                sub_near_label = "; ".join(str(g.value(np, RDFS.label) or short(np)) for np in sub_near) if sub_near else ""
                sub_near_qid = ""
                if sub_near:
                    sqids = []
                    for np in sub_near:
                        for q in g.objects(np, URIRef("http://www.w3.org/2004/02/skos/core#closeMatch")):
                            sqids.append(wd(q))
                    sub_near_qid = "; ".join(sqids)
                print(f'{step},"  {sub_id}","{ev_id}","{sub_type}","{sub_label}","","","","","{at_l}","{at_q}","{w_l}","{w_q}","{sub_near_label}","{sub_near_qid}","","{sub_cits}"')

        # Mentioned-only toponyms (E53 / pure E18 only — see is_toponym())
        for mp in sorted(mentioned_toponyms, key=str):
            label = str(g.value(mp, RDFS.label) or short(mp))
            qid = ""
            for q in g.objects(mp, URIRef("http://www.w3.org/2004/02/skos/core#closeMatch")):
                qid = wd(q)
            print(f',"","","mentioned_only","{label}","","{qid}","","","","","","",""')

        return

    # --- Human-readable mode ---
    # Header
    print("=" * 80)
    header_label = str(g.value(letter_uri, RDFS.label) or "Travelogue")
    print(f"  {header_label}")
    print(f"  Source: {Path(ttl_path).name}")
    print("=" * 80)

    # Summary line for journey
    if journey_root:
        journey_label = str(g.value(journey_root, RDFS.label) or "")
        if journey_label:
            print(f"  Journey: {journey_label}")

    # Route overview: first departure → final arrival
    top_events_list = [ev for ev in chain if ev not in skip_in_chain]
    if top_events_list:
        first_ev = top_events_list[0]
        last_ev = top_events_list[-1]
        first_from = g.value(first_ev, P27)
        # Last event may be a stay/tour — use P26 (move_to), P7 (took_place_at),
        # or P8 (took_place_on_or_within) whichever is available
        last_dest = (g.value(last_ev, P26) or g.value(last_ev, P7)
                     or g.value(last_ev, P8))
        if first_from and last_dest:
            first_label = place_repr(first_from)
            last_label = place_repr(last_dest)
            n_trans = sum(1 for ev in chain if (ev, RDF.type, E9_Move) in g)
            print(f"  Route: {first_label} → {last_label} "
                  f"({len(top_events_list)} events, {n_trans} translocations)")
    print()

    # Event chain
    step = 0
    for ev in chain:
        if ev in skip_in_chain:
            continue
        step += 1
        ev_id = short(ev)
        label = str(g.value(ev, RDFS.label) or "")
        is_move = (ev, RDF.type, E9_Move) in g

        # Determine type and details
        mfrom = g.value(ev, P27)
        mto = g.value(ev, P26)
        at_p = g.value(ev, P7)
        within = g.value(ev, P8)

        if is_move:
            etype = "TRANSLOCATION"
        elif within:
            etype = "INDOOR TOUR"
        elif at_p:
            etype = "OUTDOOR / STAY"
        else:
            etype = "STAY"

        # Build annotation badges
        badges = []
        n_subs = len(sub_events_of.get(ev, []))
        if n_subs:
            badges.append(f"{n_subs} sub-events")
        n_cits = len(get_citations(ev))
        if n_cits:
            badges.append(f"{n_cits} citations")
        badge_str = f"  ({'; '.join(badges)})" if badges else ""

        print(f"{step:>2}. [{ev_id}]  {etype}{badge_str}")
        print(f"    {label}")

        # From / To / Mode for translocations
        if is_move:
            from_str = place_repr(mfrom)
            to_str = place_repr(mto)
            print(f"    From: {from_str}")
            print(f"    To:   {to_str}")
            mode = get_mode(ev)
            if not mode and " met " in label:
                mode = label.split(" met ")[-1].rstrip(".")
            if mode:
                print(f"    Mode: {mode}")
        elif within:
            print(f"    At: {place_repr(within)}")
        elif at_p:
            print(f"    At: {place_repr(at_p)}")

        # Near places
        near_places = list(g.objects(ev, PROXIMITY))
        if near_places:
            near_strs = [place_repr(np) for np in near_places]
            print(f"    Near: {', '.join(near_strs)}")

        # Time spans
        spans = get_time_spans(ev)
        if spans:
            print(f"    Time: {', '.join(spans)}")

        # Citations
        cits = get_citations(ev)
        if cits:
            print(f"    Citations: {', '.join(mid for mid, _, _ in cits)}")

        # Sub-events
        for sub in sub_events_of.get(ev, []):
            sub_id = short(sub)
            sub_label = str(g.value(sub, RDFS.label) or "")
            sub_at = g.value(sub, P7)
            sub_within = g.value(sub, P8)
            if sub_within:
                sub_detail = place_repr(sub_within)
            elif sub_at:
                sub_detail = place_repr(sub_at)
            else:
                sub_detail = ""
            print(f"    |-- [{sub_id}] {sub_label}")
            if sub_detail:
                print(f"    |   At: {sub_detail}")
            sub_near = list(g.objects(sub, PROXIMITY))
            if sub_near:
                sub_near_strs = [place_repr(np) for np in sub_near]
                print(f"    |   Near: {', '.join(sub_near_strs)}")
            sub_cits = get_citations(sub)
            if sub_cits:
                print(f"    |   Citations: {', '.join(mid for mid, _, _ in sub_cits)}")

        print()

    # --- Places visited ---
    # place_roles / place_event_counts / visited_toponyms were computed in the
    # single shared pass above (filtered to visitable toponyms via is_toponym()).

    if place_roles:
        # Order places: departure first, then arrivals in chain order, then rest
        ordered = []
        seen = set()
        for ev in chain:
            if ev in skip_in_chain:
                continue
            if (ev, RDF.type, E9_Move) in g:
                mfrom = g.value(ev, P27)
                mto = g.value(ev, P26)
                if mfrom and is_toponym(mfrom) and mfrom not in seen:
                    ordered.append(mfrom)
                    seen.add(mfrom)
                if mto and is_toponym(mto) and mto not in seen:
                    ordered.append(mto)
                    seen.add(mto)
            else:
                at_p = g.value(ev, P7)
                within = g.value(ev, P8)
                for p in (at_p, within):
                    if p and is_toponym(p) and p not in seen:
                        ordered.append(p)
                        seen.add(p)
        # Add any remaining (e.g. sub-event places not yet seen)
        for p in place_roles:
            if p not in seen:
                ordered.append(p)

        print("-" * 80)
        print("  Places visited:")
        for p in ordered:
            label = place_repr(p)
            roles = ", ".join(sorted(place_roles.get(p, set())))
            count = place_event_counts.get(p, 0)
            count_str = f" ({count} events)" if count > 1 else ""
            print(f"    {label}  — {roles}{count_str}")
        print()

    # Mentioned-only toponyms (E53 / pure E18 only — see is_toponym())
    if mentioned_toponyms:
        print("-" * 80)
        print("  Places mentioned but not visited:")
        for mp in sorted(mentioned_toponyms, key=str):
            mp_label = str(g.value(mp, RDFS.label) or short(mp))
            mp_qid = ""
            for q in g.objects(mp, URIRef("http://www.w3.org/2004/02/skos/core#closeMatch")):
                mp_qid = f" ({wd(q)})"
            print(f"    - {mp_label}{mp_qid}")

    # Summary
    print()
    print("=" * 80)
    top_events = sum(1 for ev in chain if ev not in skip_in_chain)
    translocations = sum(1 for ev in chain if (ev, RDF.type, E9_Move) in g)
    tours = sum(1 for ev in chain if (ev, RDF.type, E9_Move) not in g)
    sub_count = len(skip_in_chain)
    ct_count = len(list(g.subjects(None, E89)))
    print(f"  Events: {top_events} top-level + {sub_count} sub-events = {top_events + sub_count} total")
    print(f"  Translocations: {translocations}")
    print(f"  Tours/Stays: {tours}")
    print(f"  CT citations: {ct_count}")
    n_near = len(list(g.subject_objects(PROXIMITY)))
    print(f"  Near places (happened_in_proximity_of): {n_near}")
    print(f"  Mentioned-only places: {len(mentioned_toponyms)}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    csv_mode = "--csv" in sys.argv

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

    query_route(path, csv_mode=csv_mode)

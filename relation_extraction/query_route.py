"""Query travel route from ATO RDF via the P183 temporal chain.

Usage:
    python relation_extraction/query_route.py [path/to/events.ttl] [--brief BRF0003]
"""

import sys
from pathlib import Path

from rdflib import RDFS, Graph
from rdflib.term import URIRef

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

CIDOC = "http://www.cidoc-crm.org/cidoc-crm/"
ATO = "http://academictourism.com/entity/"

P183 = URIRef(CIDOC + "P183_ends_before_the_start_of")
P26 = URIRef(CIDOC + "P26_moved_to")
P27 = URIRef(CIDOC + "P27_moved_from")
P7 = URIRef(CIDOC + "P7_took_place_at")
P8 = URIRef(CIDOC + "P8_took_place_on_or_within")
P2 = URIRef(CIDOC + "P2_has_type")


def query_route(ttl_path):
    g = Graph()
    g.parse(ttl_path, format="turtle")

    # Find all P183 pairs and build the chain
    pairs = list(g.subject_objects(P183))
    sources = {a for a, _ in pairs}
    targets = {b for _, b in pairs}
    first_events = sources - targets

    if not first_events:
        print("ERROR: no first event found (P183 chain has cycle or is empty)")
        return

    if len(first_events) > 1:
        print(f"WARNING: {len(first_events)} first events found, picking one")

    # Follow the chain
    chain = []
    current = sorted(first_events)[0]
    while current:
        chain.append(current)
        next_events = [b for a, b in pairs if a == current]
        current = next_events[0] if len(next_events) == 1 else None

    # Print route
    print(f"{'#':<4} {'Event':<24} {'Type':<15} {'From':<18} {'To':<18} {'At':<18} {'Within':<24} {'Label'}")
    print("-" * 150)

    for i, ev in enumerate(chain, 1):
        short_id = str(ev).replace(ATO, "")
        label = str(g.value(ev, RDFS.label) or "")
        mfrom = g.value(ev, P27)
        mto = g.value(ev, P26)
        at = g.value(ev, P7)
        within = g.value(ev, P8)

        from_label = str(g.value(mfrom, RDFS.label) or "") if mfrom else ""
        to_label = str(g.value(mto, RDFS.label) or "") if mto else ""
        at_label = str(g.value(at, RDFS.label) or "") if at else ""
        within_label = str(g.value(within, RDFS.label) or "") if within else ""

        # Determine type
        if mfrom and mto:
            etype = "translocation"
        elif within:
            etype = "indoor_tour"
        elif at:
            etype = "outdoor_tour / stay"
        else:
            etype = ""

        print(f"{i:<4} {short_id:<24} {etype:<15} {from_label:<18} {to_label:<18} {at_label:<18} {within_label:<24} {label}")

    print("-" * 150)
    print(f"Total: {len(chain)} events in chronological order")

    # Also output CSV version to stdout
    if "--csv" in sys.argv:
        print("\n--- CSV ---")
        print("step,event_id,type,from,to,at,within,label")
        for i, ev in enumerate(chain, 1):
            short_id = str(ev).replace(ATO, "")
            label = str(g.value(ev, RDFS.label) or "").replace('"', '""')
            mfrom = g.value(ev, P27)
            mto = g.value(ev, P26)
            at = g.value(ev, P7)
            within = g.value(ev, P8)
            from_label = str(g.value(mfrom, RDFS.label) or "") if mfrom else ""
            to_label = str(g.value(mto, RDFS.label) or "") if mto else ""
            at_label = str(g.value(at, RDFS.label) or "") if at else ""
            within_label = str(g.value(within, RDFS.label) or "") if within else ""
            if mfrom and mto:
                etype = "translocation"
            elif within:
                etype = "indoor_tour"
            elif at:
                etype = "outdoor_tour/stay"
            else:
                etype = ""
            print(f'{i},"{short_id}","{etype}","{from_label}","{to_label}","{at_label}","{within_label}","{label}"')


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
        else str(ROOT_DIR / "output" / "rdf" / "1816_third_letter_events.ttl")
    if not Path(path).exists():
        print(f"File not found: {path}")
        sys.exit(1)
    query_route(path)

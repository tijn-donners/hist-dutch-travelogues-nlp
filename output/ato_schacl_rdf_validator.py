"""Validate ATO RDF output against SHACL shapes.

Loads the generated events RDF (*_events.ttl) and the ATO SHACL shapes graph
(ato_schacl_shapes.ttl), then runs pyshacl validation to check compliance with
the CIDOC CRM and ATO ontology constraints.

Usage:
    python output/ato_schacl_rdf_validator.py [path/to/events.ttl]

Requires: pyshacl (pip install pyshacl)
"""

import sys
from pathlib import Path

from rdflib import Graph

SCRIPT_DIR = Path(__file__).resolve().parent
RDF_DIR = SCRIPT_DIR / "rdf"

DATA_FILE = RDF_DIR / "1816_third_letter_events.ttl"
SHAPES_FILE = RDF_DIR / "ato_schacl_shapes.ttl"

# Ontology file for additional class/property definitions needed during
# validation (the shapes file references these but doesn't always declare them)
ATO_ONTOLOGY = RDF_DIR.parent.parent / "ATO.rdf"


def select_data_file():
    """Scan rdf/ for *_events.ttl files and let the user pick one."""
    ttl_files = sorted(RDF_DIR.glob("*_events.ttl"))
    if not ttl_files:
        print(f"No *_events.ttl files found in {RDF_DIR}")
        raise SystemExit(1)

    if len(ttl_files) == 1:
        print(f"Auto-selected data: {ttl_files[0].name}")
        return str(ttl_files[0])

    print("Available data files:")
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


def validate(data_path, shapes_path, ontology_path=None):
    """Run SHACL validation on the data graph.

    Args:
        data_path: Path to the instance data TTL file.
        shapes_path: Path to the SHACL shapes TTL file.
        ontology_path: Optional path to the ontology RDF for class/property
                       definitions that the shapes reference.

    Returns:
        Tuple of (conforms: bool, results_graph: Graph, report_text: str).
    """
    from pyshacl import validate as shacl_validate

    # Load data graph
    data_graph = Graph()
    data_graph.parse(data_path, format="turtle")
    print(f"Data graph: {len(data_graph)} triples from {Path(data_path).name}")

    # Load shapes graph
    shapes_graph = Graph()
    shapes_graph.parse(shapes_path, format="turtle")
    print(f"Shapes graph: {len(shapes_graph)} triples from {Path(shapes_path).name}")

    # Load ontology for additional class/property definitions if provided
    if ontology_path and Path(ontology_path).exists():
        shapes_graph.parse(ontology_path, format="xml")
        print(f"Ontology: {len(shapes_graph)} triples total (with ontology)")

    # Repair incomplete shapes: remove PropertyShapes missing sh:path
    from rdflib.namespace import RDF, SH
    bad_shapes = []
    for ps in list(shapes_graph.subjects(RDF.type, SH.PropertyShape)):
        if not shapes_graph.value(ps, SH.path):
            bad_shapes.append(ps)
    if bad_shapes:
        print(f"  Removing {len(bad_shapes)} incomplete PropertyShape(s) "
              f"(missing sh:path)")
        for ps in bad_shapes:
            shapes_graph.remove((ps, None, None))

    print(f"\nRunning SHACL validation...")
    conforms, results_graph, report_text = shacl_validate(
        data_graph,
        shacl_graph=shapes_graph,
        ont_graph=None,
        inference="none",
    )

    return conforms, results_graph, report_text


def print_results(conforms, results_graph, report_text):
    """Print validation results in a readable format."""
    from rdflib.namespace import RDF, SH

    print()
    print("=" * 70)
    if conforms:
        print("  RESULT: CONFORMS — all shapes passed")
    else:
        print("  RESULT: VIOLATIONS FOUND")
    print("=" * 70)

    # Count by severity
    violations = list(results_graph.subjects(RDF.type, SH.ValidationResult))
    infos = list(results_graph.subjects(RDF.type, SH.Info))
    warnings = list(results_graph.subjects(RDF.type, SH.Warning))

    sev_count = {"sh:Violation": len(violations), "sh:Info": len(infos)}
    if warnings:
        sev_count["sh:Warning"] = len(warnings)

    print(f"\n  Results by severity: {sev_count}")
    print(f"  Total results: {len(violations) + len(infos) + len(warnings)}")

    if not violations and not warnings:
        print("\n  No violations or warnings. Data is SHACL-compliant.")
        return

    # Group results by source shape
    from collections import defaultdict
    grouped = defaultdict(list)

    for vr in violations + warnings:
        focus = results_graph.value(vr, SH.focusNode)
        path = results_graph.value(vr, SH.resultPath)
        message = results_graph.value(vr, SH.resultMessage)
        severity = results_graph.value(vr, SH.resultSeverity)
        source_shape = results_graph.value(vr, SH.sourceShape)
        source_constraint = results_graph.value(vr, SH.sourceConstraintComponent)

        sev_str = str(severity).replace(str(SH), "sh:") if severity else "?"

        # Shorten URIs for display
        def short(uri):
            s = str(uri)
            for prefix, ns in [
                ("ato:", "http://academictourism.com/entity/"),
                ("crm:", "http://www.cidoc-crm.org/cidoc-crm/"),
                ("wd:", "https://www.wikidata.org/wiki/"),
                ("sh:", str(SH)),
            ]:
                if s.startswith(ns):
                    return prefix + s[len(ns):]
            return s.rsplit("/", 1)[-1] if "/" in s else s

        key = short(source_shape) if source_shape else "unknown"
        grouped[key].append({
            "focus": short(focus) if focus else "?",
            "path": short(path) if path else "",
            "message": str(message) if message else "",
            "severity": sev_str,
            "constraint": short(source_constraint) if source_constraint else "",
        })

    # Print grouped results
    for shape, results in sorted(grouped.items()):
        print(f"\n  ── Shape: {shape} ({len(results)} results) ──")
        for r in results[:5]:  # Show first 5 per shape
            print(f"    [{r['severity']}] {r['focus']}")
            if r["path"]:
                print(f"      Path: {r['path']}")
            if r["message"]:
                msg = r["message"].replace("\n", " ")
                print(f"      {msg[:200]}")
        if len(results) > 5:
            print(f"    ... and {len(results) - 5} more")


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else select_data_file()

    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)
    if not SHAPES_FILE.exists():
        print(f"Shapes file not found: {SHAPES_FILE}")
        sys.exit(1)

    print("=" * 70)
    print("  ATO RDF SHACL Validator")
    print("=" * 70)
    print(f"  Data:   {data_path}")
    print(f"  Shapes: {SHAPES_FILE}")
    print()

    try:
        conforms, results_graph, report_text = validate(
            data_path, str(SHAPES_FILE),
            ontology_path=str(ATO_ONTOLOGY) if ATO_ONTOLOGY.exists() else None,
        )
    except ImportError:
        print("ERROR: pyshacl is not installed.")
        print("Install it with: pip install pyshacl")
        sys.exit(1)

    print_results(conforms, results_graph, report_text)


if __name__ == "__main__":
    main()

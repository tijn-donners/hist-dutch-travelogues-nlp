"""Visualize NER/EL results from a .spacy DocBin in the browser.

Loads a saved DocBin, attaches kb_ids as span extensions for Wikidata links,
and launches a displaCy server with color-coded entity labels.
"""

import sys
from pathlib import Path

from spacy import displacy
import spacy
from spacy.tokens import Doc, DocBin, Span, Token

# Register custom extension attributes for entities and tokens (must match those in el.py)
if not Span.has_extension("kb_id_wikidata_"):
    Span.set_extension("kb_id_wikidata_", default=None)
if not Token.has_extension("ent_kb_id_wikidata_"):
    Token.set_extension("ent_kb_id_wikidata_", default=None)
if not Span.has_extension("kb_id_geonames_"):
    Span.set_extension("kb_id_geonames_", default=None)
if not Token.has_extension("ent_kb_id_geonames_"):
    Token.set_extension("ent_kb_id_geonames_", default=None)


def main():
    """Load a user-selected .spacy file and serve displaCy visualization."""
    ROOT_DIR = Path(__file__).resolve().parent.parent
    RESULTS_NER = ROOT_DIR / "ner" / "ner-output"
    RESULTS_EL = ROOT_DIR / "entity_linking" / "el-results"
    spacy_files = sorted(RESULTS_NER.rglob("*.spacy")) + sorted(RESULTS_EL.glob("*.spacy" ))

    if not spacy_files:
        print("No .spacy files found in ner-output/")
        sys.exit(1)

    print("Available .spacy files:")
    for i, f in enumerate(spacy_files, 1):
        # Show relative path if under ner-output, else just filename
        try:
            label = f.relative_to(RESULTS_NER)
        except ValueError:
            label = f.name
        print(f"  {i:>3}. {label} ")

    try:
        choice = input(f"\nSelect a file (1-{len(spacy_files)}): ").strip()
        idx = int(choice) - 1
        if idx < 0 or idx >= len(spacy_files):
            raise ValueError
    except (ValueError, EOFError, KeyboardInterrupt):
        print("Invalid selection.")
        sys.exit(1)

    FILE = str(spacy_files[idx])
    print(f"\nLoading: {FILE}\n")

    nlp = spacy.blank("nl")
    db = DocBin().from_disk(FILE)
    docs = list(db.get_docs(nlp.vocab))

    visualized_docs = []

    for doc in docs:
        new_ents = []
        for ent in doc.ents:
            # Get both KB IDs from the entity (using extension attributes)
            try:
                wikidata_id = ent._.kb_id_wikidata_
            except (AttributeError, KeyError):
                wikidata_id = None
            try:
                geonames_id = ent._.kb_id_geonames_
            except (AttributeError, KeyError):
                geonames_id = None

            # For backward compatibility, define a primary kb_id (prefer Wikidata if available)
            kb_id = wikidata_id if wikidata_id is not None else geonames_id

            # Create label that includes both IDs if available
            label_parts = [ent.label_]
            if wikidata_id is not None:
                label_parts.append(f"Wikidata:{wikidata_id}")
            if geonames_id is not None:
                label_parts.append(f"GeoNames:{geonames_id}")
            label = " ".join(label_parts)

            new_span = spacy.tokens.Span(doc, ent.start, ent.end, label=label)
            # Set kb_id_ to Wikidata ID for the Wikidata link (if available), otherwise GeoNames
            new_span.kb_id_ = kb_id
            new_ents.append(new_span)

        display_doc = Doc(doc.vocab, words=[t.text for t in doc], spaces=[t.whitespace_ for t in doc])
        display_doc.ents = new_ents
        visualized_docs.append(display_doc)

    colors = {
        "E53_Place": "#cce5ff",
        "E18_Physical_Thing": "#d4edda",
        "Mode_of_Transportation": "#fff3cd",
        "E52_Time_Span": "#f8d7da",
        "F2_Expression": "#e2d1f0",
        "E19_Physical_Object": "#cce5cc",
        "E20_Biological_Object": "#b3d9b3",
        "E31_Document": "#ffe0cc",
    }

    displacy.serve(
        visualized_docs,
        style="ent",
        options={
            "colors": colors,
            "kb_url_template": "https://www.wikidata.org/wiki/{}",
        },
    )


if __name__ == "__main__":
    main()

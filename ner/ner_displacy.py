"""Visualize NER/EL results from a .spacy DocBin in the browser.

Loads a saved DocBin, attaches kb_ids as span extensions for Wikidata links,
and launches a displaCy server with color-coded entity labels.
"""

import sys
from pathlib import Path

from spacy import displacy
import spacy
from spacy.tokens import Doc, DocBin


def main():
    """Load a user-selected .spacy file and serve displaCy visualization."""
    SCRIPT_DIR = Path(__file__).resolve().parent
    RESULTS_DIR = SCRIPT_DIR / "ner-results"
    spacy_files = sorted(RESULTS_DIR.glob("*.spacy"))

    if not spacy_files:
        print("No .spacy files found in ner-results/")
        sys.exit(1)

    print("Available .spacy files:")
    for i, f in enumerate(spacy_files, 1):
        print(f"  {i:>3}. {f.name} ")

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
            kb_id = ent.kb_id_
            new_span = spacy.tokens.Span(doc, ent.start, ent.end, label=ent.label_)
            new_span.kb_id_ = kb_id
            new_ents.append(new_span)

        display_doc = Doc(doc.vocab, words=[t.text for t in doc], spaces=[t.whitespace_ for t in doc])
        display_doc.ents = new_ents
        visualized_docs.append(display_doc)

    colors = {
        "E53_Place": "#cce5ff",
        "E19_Physical_Thing": "#d4edda",
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

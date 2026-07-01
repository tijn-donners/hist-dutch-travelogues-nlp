"""Alias → CIDOC label mapping for the GLiNER v1 multilingual baseline.

GLiNER v1 takes a *flat list of label strings* (not the `{label: description}`
schema that GLiNER2 used), and it performs best with natural-language labels.
So we feed readable English aliases and map each prediction back to the
canonical CIDOC-CRM label the evaluation contract expects (the same 8 labels
defined in ``ner/config/fewshot.cfg`` -> ``[components.llm.task.label_definitions]``).

Keep these aliases / mappings in sync with the .cfg ontology if it changes.
The alias text mirrors the intent of each CIDOC definition there.
"""

# readable alias -> canonical CIDOC label (must match ner_evaluate.py expectations)
ALIAS_TO_CIDOC = {
    "place or location": "E53_Place",
    "building or physical structure": "E18_Physical_Thing",
    "mode of transportation": "Mode_of_Transportation",
    "time or date": "E52_Time_Span",
    "artistic or creative expression": "F2_Expression",
    "movable physical object": "E19_Physical_Object",
    "biological object": "E20_Biological_Object",
    "specific document, book or publication": "E31_Document",
}

# canonical CIDOC labels (for validation / reference)
CIDOC_LABELS = list(dict.fromkeys(ALIAS_TO_CIDOC.values()))
# relation_extraction/

Stage 3: Relation Extraction and RDF generation. Takes gold NER+EL output (`*_el.spacy`) and asks an LLM to classify each place mention's relation to the travel events per the Academic Tourism Ontology (ATO). Categories are `IN`, `NEAR`, `THRU`, `TO`, `FROM`, or `NO_REL`. The module then builds travel events (`E9_Move`, `E7_Activity`) with their place roles and emits RDF (`.ttl`). Evaluation compares the LLM's per-entity classification against the gold CSV (with Soni et al. categories) for both the JSON output and the RDF triples.

## Files

- `rel_extraction.py`: main RE pipeline. See the root README for full CLI flags.
- `re_evaluate.py`: evaluate RE toponym-relation classification against gold.
- `validate_rdf.py`: validate generated RDF against ATO/CIDOC-CRM schema constraints.
- `query_route.py`: query and display the chronological travel route from the ATO RDF (supports `--csv`).
- `run_re_gold_models.py`: batch harness that runs every NER-stage model through RE on gold EL output, evaluates each, and appends to `re-evaluation/scores.csv`. Idempotent and resumable.
- `trace_citation_coverage.py`: diagnostic that traces how faithfully the RDF reflects the LLM's per-entity classification.
- `re_prompt_dutch.jinja`, `re_examples.yaml`: RE prompt template and few-shot examples.
- `1816_re_gs.csv`: gold-standard RE CSV.
- `re-evaluation/`: `scores.csv` plus per-run `re_eval_*.json` metadata.
- `figures/`: confusion matrices and heatmaps.
- `re_error_analysis.ipynb`, `re_run_comparison.ipynb`: analysis notebooks.

## Running

```
python relation_extraction/rel_extraction.py --model gemma4:31b \
  --temperature 0.0 --input entity_linking/el-results/<file>_el.spacy --host cloud
```

If `--input` is omitted, an interactive picker lists `*_el.spacy` files. Output (events JSON, mention map, reasoning trace, and `.ttl` RDF) goes to `output/re/` and `output/rdf/`.

Evaluate:

```
python relation_extraction/re_evaluate.py --events output/re/<file>_events.json --rdf output/rdf/<file>_events.ttl
```

See the root README for the full flag table.

## CSV columns

### `1816_re_gs.csv`

Reference gold standard for RE evaluation: the EL gold fields plus spatial-relation categories.

| Column | Description |
|---|---|
| `text` | Mention text as written in the letter |
| `label` | CIDOC-CRM entity label |
| `start_char` | Start character offset in the source text |
| `end_char` | End character offset in the source text |
| `wikidata_qid` | Wikidata QID |
| `geonames_id` | GeoNames ID |
| `note` | Free-text note |
| `visited_type` | Soni et al. spatial-relation category: `IN`, `NEAR`, `THRU`, `TO`, `FROM`, or `NO_REL` |
| `visited_type_note` | Free-text justification for the category |

### `re-evaluation/scores.csv`

One RE evaluation result per model run.

| Column | Description |
|---|---|
| `source_text` | Source text identifier |
| `model` | Ollama model name |
| `temperature` | LLM temperature |
| `think_mode` | Thinking mode used |
| `inference_type` | `cloud` or `local` |
| `datetime` | Run timestamp |
| `duration_seconds` | Run duration |
| `fine_macro_f1` | Macro F1 across relation types |
| `fine_micro_f1` | Micro F1 across relation types |
| `thru_f1` | F1 for the THRU relation |
| `to_f1` | F1 for the TO relation |
| `from_f1` | F1 for the FROM relation |
| `in_f1` | F1 for the IN relation |
| `near_f1` | F1 for the NEAR relation |
| `no_rel_f1` | F1 for the NO_REL category |
| `binary_json_p` | Binary visited-vs-not precision (JSON) |
| `binary_json_r` | Binary recall (JSON) |
| `binary_json_f1` | Binary F1 (JSON) |
| `binary_json_accuracy` | Binary accuracy (JSON) |
| `binary_rdf_p` | Binary precision (RDF) |
| `binary_rdf_r` | Binary recall (RDF) |
| `binary_rdf_f1` | Binary F1 (RDF) |
| `binary_rdf_accuracy` | Binary accuracy (RDF) |
| `rdf_entities` | Entities written to RDF |
| `rdf_dropped` | Entities dropped from RDF |
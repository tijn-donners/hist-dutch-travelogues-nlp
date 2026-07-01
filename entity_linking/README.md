# entity_linking/

Stage 2: Entity Linking. Takes NER `.spacy` output and resolves each mention to a Wikidata QID and/or GeoNames ID, or NIL when nothing fits. The pipeline is a three-stage LELA-inspired architecture:

1. Candidate generation (`el_candidates.py`): an LLM predicts modern name/coordinates/region from the archaic Dutch context, then queries Wikidata and a GeoNames Germany gazetteer for semantic candidate matching.
2. Reranking (`el_reranker.py`): pointwise Ollama LLM scoring of candidates against context.
3. LLM selection (`el_selector.py`): the LLM picks the best KB candidate (enriched with Wikipedia extracts) or returns NIL.

Evaluation scores Wikidata-QID and GeoNames-ID prediction against a gold CSV, both EL-only (conditioned on overlapping spans) and end-to-end.

## Files

- `el.py`: main EL pipeline. See the root README for full CLI flags.
- `el_candidates.py`, `el_reranker.py`, `el_selector.py`: stages 1, 2, 3.
- `el_evaluate.py`: evaluate predictions against gold KB IDs.
- `el_from_gs.py`, `build_gs_spacy.py`: build gold-standard `.spacy` files for the pipeline.
- `el_stats.py`: print entity-linking statistics for an `_el.spacy` file.
- `batch_evaluate.py`: batch-evaluate every `*_el.spacy` in `el-results/`.
- `run_el_configs.py`: run all unfinished EL configs for one model.
- `build_legacy_thinking_scores.py`, `probe_think_modes.py`: thinking-mode analysis and legacy-run salvage.
- `GeoNames_DE_gazetteer.txt`: German GeoNames gazetteer (tab-separated).
- `1816_el_gs.csv`: gold-standard EL CSV.
- `run_el_eval.sh`: shell driver for EL evaluation.
- `el_metrics_notebook.ipynb`, `el_metrics_notebook_legacy.ipynb`: analysis notebooks.
- `el-results/`, `el-results-bugged-run/`, `el-evaluation/`, `qrank/`, `figures/`: output and data subfolders. See `qrank/README.md` for QRank analysis.

## Running

```
python entity_linking/el.py --model gemma4:31b --temperature 0.0 --top-k 3 \
  --input ner/ner-output/<model>/1816/<file>.spacy --host cloud
```

If `--input` is omitted, an interactive picker lists available `.spacy` files. Output goes to `el-results/`.

Evaluate:

```
python entity_linking/el_evaluate.py --el el-results/<file>_el.spacy --gold 1816_el_gs.csv
```

See the root README for the full flag table.

## CSV columns

### `1816_el_gs.csv`

Reference gold standard for all EL evaluation.

| Column | Description |
|---|---|
| `text` | Mention text as written in the letter |
| `label` | CIDOC-CRM entity label |
| `start_char` | Start character offset in the source text |
| `end_char` | End character offset in the source text |
| `wikidata_qid` | Manually verified Wikidata QID |
| `geonames_id` | Manually verified GeoNames ID |
| `note` | Free-text note |

### `el-evaluation/scores.csv`

One EL evaluation result per model run.

| Column | Description |
|---|---|
| `source_text` | Source text identifier |
| `model` | Ollama model name |
| `temperature` | LLM temperature |
| `top_k_rerank` | Top-k rerank value |
| `think` | Thinking mode used |
| `inference_type` | `cloud` or `local` |
| `total_entities` | Total entities evaluated |
| `duration_seconds` | Run duration |
| `datetime` | Run timestamp |
| `count_errors` | Number of errored entities |
| `count_skipped` | Number of skipped entities |
| `count_linked_wd` | Entities linked to Wikidata |
| `count_linked_gn` | Entities linked to GeoNames |
| `count_linked_both` | Entities linked to both |
| `count_linked_neither` | Entities linked to neither |
| `wd_p` | Wikidata precision |
| `wd_r` | Wikidata recall |
| `wd_f1` | Wikidata F1 |
| `gn_p` | GeoNames precision |
| `gn_r` | GeoNames recall |
| `gn_f1` | GeoNames F1 |
| `combined_p` | Combined precision |
| `combined_r` | Combined recall |
| `combined_f1` | Combined F1 |
| `nil_p` | NIL precision |
| `nil_r` | NIL recall |
| `nil_f1` | NIL F1 |
| `nil_wd_p` | NIL-vs-Wikidata precision |
| `nil_wd_r` | NIL-vs-Wikidata recall |
| `nil_wd_f1` | NIL-vs-Wikidata F1 |
| `nil_gn_p` | NIL-vs-GeoNames precision |
| `nil_gn_r` | NIL-vs-GeoNames recall |
| `nil_gn_f1` | NIL-vs-GeoNames F1 |

### `el-evaluation/scores_legacy_thinking.csv`

Salvaged metrics for the legacy thinking-mode runs (low/medium/high) from `el-results-bugged-run/`, with corrected temperature bookkeeping. Same 32 columns as `scores.csv`.

### `el-results/*_el_el_eval.csv` and `el-results-bugged-run/*_el_el_eval.csv`

Per-entity gold-vs-predicted comparison per page. (Note: these files have a leading UTF-8 BOM.)

| Column | Description |
|---|---|
| `page` | Page number |
| `type_combined` | Match type (TP/FP/FN) for combined WD+GN |
| `type_wd` | Match type for Wikidata only |
| `type_gn` | Match type for GeoNames only |
| `type_nil_wd` | NIL-vs-Wikidata match type |
| `type_nil_gn` | NIL-vs-GeoNames match type |
| `type_nil_combined` | NIL-combined match type |
| `kb` | Knowledge base |
| `gold_text` | Gold mention text |
| `pred_text` | Predicted mention text |
| `gold_label` | Gold CIDOC label |
| `gold_wikidata` | Gold Wikidata QID |
| `gold_geonames` | Gold GeoNames ID |
| `pred_wikidata` | Predicted Wikidata QID |
| `pred_geonames` | Predicted GeoNames ID |
| `note` | Free-text note |

### `qrank/1816_qrank_per_entity.csv`

One row per unique Wikidata entity in the gold set, with its QRank popularity.

| Column | Description |
|---|---|
| `wikidata_qid` | Wikidata QID |
| `wikidata_label` | English Wikidata label |
| `text` | Mention text(s) as written in the letter |
| `label` | CIDOC label |
| `QRank` | QRank popularity score |
| `n_mentions` | Number of mentions in the text |
| `in_qrank` | Whether the entity appears in the QRank dump (True/False) |

### `qrank/1816_qrank_matched.csv`

Per-mention view of QRank coverage.

| Column | Description |
|---|---|
| `text` | Mention text |
| `label` | CIDOC label |
| `wikidata_qid` | Wikidata QID |
| `wikidata_label` | English Wikidata label |
| `geonames_id` | GeoNames ID |
| `QRank` | QRank popularity score |
| `in_qrank` | Whether the entity appears in the QRank dump (True/False) |
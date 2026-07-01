# el-results-bugged-run/

Legacy Entity Linking results that ran before the temperature chain bug was fixed.

The most important runs are the ones where thinking mode was enabled, but the slug filenames and metadata are wrong:

- The EPG in the candidate generation stage ran at 0.1 temperature.
- The reranker and selector LLMs ran at 0.0.

The rest of the metadata is still valid.

This folder preserves these results because inference time for the thinking modes was substantial. They can be used for brief analysis of F1 performance when thinking is enabled, compared to non-thinking mode. Use `build_legacy_thinking_scores.py` (in `entity_linking/`) to correct the bookkeeping (rename `_t1.0_` to `_t0.1_`, fix `*_run_info.json`, rebuild `scores_legacy_thinking.csv`) without re-inference.

## CSV columns

The `*_el_el_eval.csv` files share one header (with a leading UTF-8 BOM):

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
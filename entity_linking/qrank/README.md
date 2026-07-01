# qrank/

Evaluates how "popular" the linked Wikidata entities in the test set are, to indicate the degree of long-tail entities in this dataset.

## Reproduce

Download the latest QRank dump from https://qrank.toolforge.org/ (~100 MB, unzipped ~381 MB) and place the unzipped `.csv` in this directory, then run the analysis in `qrank_analysis.ipynb`.

## Files

- `1816_qrank_per_entity.csv`: per-entity QRank view (one row per unique Wikidata entity).
- `1816_qrank_matched.csv`: per-mention QRank view (one row per gold mention matched to its Wikidata entity).
- `qid_labels.json`: cached Wikidata labels.
- `qrank_analysis.ipynb`: analysis notebook.

## CSV columns

### `1816_qrank_per_entity.csv`

| Column | Description |
|---|---|
| `wikidata_qid` | Wikidata QID |
| `wikidata_label` | English Wikidata label |
| `text` | Mention text(s) as written in the letter |
| `label` | CIDOC label |
| `QRank` | QRank popularity score |
| `n_mentions` | Number of mentions in the text |
| `in_qrank` | Whether the entity appears in the QRank dump (True/False) |

### `1816_qrank_matched.csv`

| Column | Description |
|---|---|
| `text` | Mention text |
| `label` | CIDOC label |
| `wikidata_qid` | Wikidata QID |
| `wikidata_label` | English Wikidata label |
| `geonames_id` | GeoNames ID |
| `QRank` | QRank popularity score |
| `in_qrank` | Whether the entity appears in the QRank dump (True/False) |
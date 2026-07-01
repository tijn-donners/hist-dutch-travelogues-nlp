# ner/

Stage 1: Named Entity Recognition. Runs spaCy-LLM with an Ollama LLM over the source letter texts to tag 8 CIDOC-CRM entity types (`E53_Place`, `E18_Physical_Thing`, `Mode_of_Transportation`, `E52_Time_Span`, `F2_Expression`, `E19_Physical_Object`, `E20_Biological_Object`, `E31_Document`). The text is split into pages by `[N]` markers, inference runs through a spaCy-LLM config, and results are saved as `.spacy` DocBins plus offset maps and meta JSON. Evaluation aligns predictions to the Recogito gold-standard annotations and reports per-label strict/relaxed P/R/F1.

## Files

- `ner.py`: main NER pipeline. See the root README for full CLI flags.
- `ner_evaluate.py`: evaluate predictions against Recogito gold annotations (`gs_annotations/1816.jsonld`).
- `ner_displacy.py`: browser visualizer of NER/EL `.spacy` files via displaCy.
- `streaming_patch.py`: monkey-patches the LangChain Ollama backend to print streaming tokens in real time.
- `export_gs_to_el_template.py`: builds the EL gold-standard CSV template from the Recogito annotations.
- `export_ner_to_el_template.py`: builds an EL annotation template from a NER `.spacy` DocBin (empty KB-ID columns for manual filling).
- `config/`: spaCy-LLM configs (`fewshot.cfg`, `fewshot_english.cfg`, `zeroshot.cfg`, `zeroshot_english.cfg`), Jinja prompts (`prompt_dutch.jinja`, `prompt_english.jinja`), and few-shot example YAMLs.
- `gs_annotations/`: gold standard (`1816.jsonld`, `gs_el_gold_template.csv`, `map_old_el_gs.py`).
- `ner-output/`: one subdir per model, each holding `.spacy`, `_offset_map.json`, `_meta.json` triples across config combinations.
- `ner-evaluation/`: `scores.csv` and `errors_logs/` (per-run error CSVs).
- `figures/`: evaluation plots. `ner_evaluation_analysis.ipynb`, `zeroshot_error_analysis.ipynb`: analysis notebooks.
- Shell scripts: `ner.sh`, `ner_and_eval.sh`, `ner_and_eval_all_fewshot.sh`, `evaluate_all.sh`, `run_missing_zeroshot.sh`.

## Running

```
python ner/ner.py --model gemma4:31b --mode fewshot --language dutch \
  --temperature 0.0 --input data/1816_third_letter.txt --ollama-host cloud
```

Evaluate:

```
python ner/ner_evaluate.py -s ner/ner-output/<model>/1816/<file>.spacy
```

See the root README for the full flag table.

## CSV columns

### `gs_annotations/gs_el_gold_template.csv`

Gold-standard entities built from the Recogito annotations. Used for NER+EL end-to-end and EL-only evaluation. (Note: this file has a leading UTF-8 BOM.)

| Column | Description |
|---|---|
| `text` | Mention text as written in the letter |
| `label` | CIDOC-CRM entity label |
| `start_char` | Start character offset in the source text |
| `end_char` | End character offset in the source text |
| `wikidata_qid` | Manually filled Wikidata QID |
| `geonames_id` | Manually filled GeoNames ID |
| `note` | Free-text note |

### `ner-evaluation/scores.csv`

One NER evaluation result per (model run x entity label).

| Column | Description |
|---|---|
| `source_text` | Source text identifier |
| `model` | Ollama model name |
| `temperature` | LLM temperature |
| `prompting_method` | `fewshot` or `zeroshot` |
| `prompt_language` | `dutch` or `english` |
| `pages_processed` | Number of pages processed |
| `duration_seconds` | Run duration |
| `inference_type` | `cloud` or `local` |
| `count_failed_pages` | Number of pages that failed |
| `datetime` | Run timestamp |
| `label` | CIDOC entity label for this row |
| `strict_p` | Strict precision |
| `strict_r` | Strict recall |
| `strict_f1` | Strict F1 |
| `relaxed_p` | Relaxed precision (overlap-based span matching) |
| `relaxed_r` | Relaxed recall |
| `relaxed_f1` | Relaxed F1 |
| `count_tp` | True positives |
| `count_fp` | False positives |
| `count_fn` | False negatives |

### `ner-evaluation/errors_logs/*.csv`

Per-run NER error logs (one file per model run, ~97 files sharing this header). Filename encodes `{source}__{model}_t{temp}_{prompting}_{language}_errors.csv`.

| Column | Description |
|---|---|
| `type` | Error type (FP, FN, label confusion) |
| `label` | CIDOC entity label |
| `page` | Page number |
| `gold_text` | Gold mention text |
| `pred_text` | Predicted mention text |
| `gold_start` | Gold start offset |
| `gold_end` | Gold end offset |
| `pred_start` | Predicted start offset |
| `pred_end` | Predicted end offset |
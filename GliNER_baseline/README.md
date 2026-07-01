# GliNER_baseline/

A non-LLM NER baseline so the LLM-based results (`ner/ner.py` via Ollama) can be contextualised against a strong off-the-shelf encoder model on identical data, identical CIDOC labels, and the identical evaluation harness (`ner/ner_evaluate.py`).

Uses [GLiNER v1](https://github.com/urchade/GLiNER) with **`urchade/gliner_multi-v2.1`** (~209M, XLM-RoBERTa, multilingual incl. Dutch). XLM-RoBERTa was pretrained on Dutch, so unlike the earlier English-only GLiNER2 baseline this encoder actually understands the language. This replaces the prior GLiNER2 baseline (`fastino/gliner2-base-v1`, English DeBERTa-v3) entirely.

## Files

- `gliner_baseline.py`: GLiNER v1 multilingual zero-shot NER. Feeds readable English label aliases and maps predictions back to the 8 canonical CIDOC labels; produces output matching `ner/ner.py`'s I/O contract so `ner/ner_evaluate.py` scores it unchanged.
- `labels.py`: alias to CIDOC label mapping (8 labels).
- `run_baseline.sh`: runs the baseline and evaluates it.
- `gliner_results.ipynb`: results notebook (per-label F1 matrix and a comparison heatmap against the LLM runs).
- `results/`: result CSVs and plots.

## Label strategy

GLiNER v1 takes a flat list of label strings (not GLiNER2's `{label: description}` schema) and performs best with natural-language labels. So `labels.py` maps readable aliases to the 8 canonical CIDOC labels:

| alias (fed to GLiNER) | CIDOC label |
|---|---|
| place or location | E53_Place |
| building or physical structure | E18_Physical_Thing |
| mode of transportation | Mode_of_Transportation |
| time or date | E52_Time_Span |
| artistic or creative expression | F2_Expression |
| movable physical object | E19_Physical_Object |
| biological object | E20_Biological_Object |
| specific document, book or publication | E31_Document |

Predictions are tagged with the canonical CIDOC label, so `ner_evaluate.py`'s per-label scoring lines up with the LLM runs.

## Install

```
pip install "gliner==0.2.21"
```

Pin `gliner==0.2.21`. Newer gliner (0.2.27) requires `transformers>=4.51.3`, which conflicts with the project's pinned `transformers==4.49.0` (required by `spacy-transformers`). 0.2.21 is compatible with transformers 4.49 and has the standard `predict_entities` API. `torch` is already in the project env; first run downloads `urchade/gliner_multi-v2.1` (~1 GB) into the HuggingFace cache.

## Hardware / feasibility

GLiNER_multi-v2.1 runs on CPU; the corpus is small (10 pages for `1816`, ~2 KB each) so a full run takes a few minutes. RAM is the binding constraint, so close heavy processes before running.

## Running

```
cd GliNER_baseline
bash run_baseline.sh                                  # defaults: 1816, brackets
bash run_baseline.sh --input data/1809_sixth_letter.txt
bash run_baseline.sh --threshold 0.4                  # lower = higher recall
```

`run_baseline.sh` runs the baseline and then evaluates it. To run only the baseline (no evaluation): `python gliner_baseline.py --no-eval`.

`gliner_baseline.py` flags: `--input/-i` (default `data/1816_third_letter.txt`), `--split-mode` (`brackets` or `horizontal-rule`, default `brackets`), `--language` (default `dutch`), `--gliner-model` (default `urchade/gliner_multi-v2.1`), `--threshold/-t` (float), `--max-len` (int, default 512), `--no-eval`.

## Output

Sidecars are written under `ner/ner-output/gliner_multi/{short_label}/` (the existing output tree, so the evaluator finds them), using the standard filename convention consumed by `ner_evaluate.py`:

```
{letter}__gliner_multi_t0.0_zeroshot_{language}.spacy
{letter}__gliner_multi_t0.0_zeroshot_{language}_offset_map.json
{letter}__gliner_multi_t0.0_zeroshot_{language}_meta.json
```

Evaluation results (per-label strict + relaxed P/R/F1) are appended to `ner/ner-evaluation/scores.csv` with `model = gliner_multi` and `inference_type = cpu`; TP/FP/FN land in `ner/ner-evaluation/errors_logs/`.

## CSV columns

### `results/gliner_f1_matrix.csv`

| Column | Description |
|---|---|
| `label` | CIDOC entity label |
| `strict_f1` | Strict F1 for this label |
| `relaxed_f1` | Relaxed F1 for this label |

### `results/gliner_per_label.csv`

| Column | Description |
|---|---|
| `label` | CIDOC entity label |
| `strict_p` | Strict precision |
| `strict_r` | Strict recall |
| `strict_f1` | Strict F1 |
| `relaxed_p` | Relaxed precision |
| `relaxed_r` | Relaxed recall |
| `relaxed_f1` | Relaxed F1 |
| `count_tp` | True positives |
| `count_fp` | False positives |
| `count_fn` | False negatives |

### `results/overall_f1_by_run.csv`

| Column | Description |
|---|---|
| `run` | Named NER run (used to compare GLiNER against the LLM runs) |
| `strict_f1` | Overall micro strict F1 |
| `relaxed_f1` | Overall micro relaxed F1 |

### `results/per_label_f1_comparison_relaxed.csv`

Per-CIDOC-label relaxed F1 across all NER runs (8 LLM models x {fewshot, zeroshot} plus the GLiNER baseline zeroshot). `label` plus one column per `model/mode` combination (e.g. `gemma4:31b/fewshot`, `gemma4:31b/zeroshot`, `gliner_multi/zeroshot`).

| Column | Description |
|---|---|
| `label` | CIDOC entity label |
| `{model}/{mode}` | Relaxed F1 for that model/mode run (one column each) |

## Caveat

GLiNER_multi-v2.1 is multilingual but not fine-tuned on historical (19th-c.) Dutch; it was trained on modern Dutch (CommonCrawl). Expect better recall than the English-GLiNER2 baseline, but still below the LLMs on archaic spellings. If recall is too low, the next step (out of scope here) is a LoRA fine-tune on the project's fewshot examples via `GLiNER.train_model(...)`.
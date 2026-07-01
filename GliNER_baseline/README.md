# GLiNER v1 multilingual baseline NER

A non-LLM baseline for the travelogue NER pipeline, so the LLM-based results
(`ner/ner.py` via Ollama) can be contextualised against a strong off-the-shelf
encoder model on **identical data, identical CIDOC labels, and the identical
evaluation harness** (`ner/ner_evaluate.py`).

Uses [GLiNER v1](https://github.com/urchade/GLiNER) with
**`urchade/gliner_multi-v2.1`** (~209M, XLM-RoBERTa, multilingual incl. Dutch).
XLM-RoBERTa was pretrained on Dutch, so — unlike the earlier English-only
GLiNER2 baseline — this encoder actually understands the language.

This **replaces** the prior GLiNER2 baseline (`fastino/gliner2-base-v1`,
English DeBERTa-v3) entirely.

## Label strategy

GLiNER v1 takes a **flat list of label strings** (not GLiNER2's
`{label: description}` schema), and performs best with natural-language labels.
So `labels.py` maps readable aliases → the 8 canonical CIDOC labels:

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

Predictions are tagged with the canonical CIDOC label, so `ner_evaluate.py`'s
per-label scoring lines up with the LLM runs.

## Install

```bash
pip install "gliner==0.2.21"
```

> **Version note:** pin `gliner==0.2.21`. Newer gliner (0.2.27) requires
> `transformers>=4.51.3`, which conflicts with the project's pinned
> `transformers==4.49.0` (required by `spacy-transformers`). 0.2.21 is
> compatible with transformers 4.49 and has the standard `predict_entities` API.
> `torch` is already in the project env; first run downloads `urchade/gliner_multi-v2.1`
> (~1 GB) into the HuggingFace cache.

## Hardware / feasibility

No GPU (CPU only), 4 cores, ~7.6 GB RAM, 15 GB disk. GLiNER_multi-v2.1 runs on
CPU; the corpus is small (10 pages for `1816`, ~2 KB each) so a full run takes a
few minutes. **RAM is the binding constraint** — close heavy processes before
running. For the large variant or faster runs, route through the existing
`SLURM/gt_to_pagexml_mapping.sh` `--partition=gpu` plumbing.

## Run

```bash
cd GliNER_baseline
bash run_baseline.sh                                   # defaults: 1816, brackets
bash run_baseline.sh --input data/1809_sixth_letter.txt
bash run_baseline.sh --threshold 0.4                    # lower = higher recall
```

`run_baseline.sh` runs the baseline and then evaluates it. To run only the
baseline (no evaluation): `python gliner_baseline.py --no-eval`.

## Output

Sidecars are written under `ner/ner-output/gliner_multi/{short_label}/` (the
existing output tree, so the evaluator finds them), using the standard filename
convention consumed by `ner_evaluate.py`:

```
{letter}__gliner_multi_t0.0_zeroshot_{language}.spacy
{letter}__gliner_multi_t0.0_zeroshot_{language}_offset_map.json
{letter}__gliner_multi_t0.0_zeroshot_{language}_meta.json
```

Evaluation results (per-label strict + relaxed P/R/F1) are appended to
`ner/ner-evaluation/scores.csv` with `model = gliner_multi`, `inference_type = cpu`,
and TP/FP/FN land in `ner/ner-evaluation/errors_logs/`. The
`gliner_results.ipynb` notebook reports and visualises them (per-label F1 matrix
+ a comparison heatmap against the LLM runs).

## Caveat

GLiNER_multi-v2.1 is multilingual but not fine-tuned on **historical** (19th-c.)
Dutch — it was trained on modern Dutch (CommonCrawl). Expect better recall than
the English-GLiNER2 baseline, but still below the LLMs on archaic spellings. If
recall is too low, the next step (out of scope here) is a LoRA fine-tune on the
project's fewshot examples via `GLiNER.train_model(...)`.
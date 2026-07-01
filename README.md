# hist-dutch-travelogues-nlp

An NLP research pipeline for 19th-century Dutch travelogues. It runs named entity recognition (NER), links entities to Wikidata and GeoNames, extracts travel-event relations, and exports the results as scholarly editions (ALTO, TEI, and RDF). All LLM inference goes through Ollama, either the Ollama cloud API or a local Ollama server.

The primary test document is the 1816 "third letter". The pipeline covers four stages:

1. HTR correction: align noisy Loghi HTR line text to a ground-truth transcription.
2. NER: tag 8 CIDOC-CRM entity types with spaCy-LLM + Ollama.
3. Entity Linking: resolve each mention to a Wikidata QID and/or GeoNames ID.
4. Relation Extraction + edition export: classify place-mention relations per the Academic Tourism Ontology (ATO), build E9_Move / E7_Activity events, and emit ALTO/TEI/RDF.

Each stage has its own subfolder with a dedicated README for details:
[`data/`](data/README.md), [`ner/`](ner/README.md), [`entity_linking/`](entity_linking/README.md), [`relation_extraction/`](relation_extraction/README.md), [`output/`](output/README.md), [`GliNER_baseline/`](GliNER_baseline/README.md), [`ground_truth_mapping/`](ground_truth_mapping/README.md), [`SLURM/`](SLURM/README.md).

## Requirements

### Python packages

Install the project dependencies (the list is unpinned in `requirements.txt`):

```
spacy, spacy-llm, ollama, httpx, rdflib, pyshacl, jinja2, pyyaml,
pandas, rapidfuzz, requests, python-dotenv, gliner
```

```
pip install -r requirements.txt
```

GLiNER (used by the baseline in `GliNER_baseline/`) needs a pinned version and torch:

```
pip install "gliner==0.2.21" torch
```

GLiNER 0.2.27 requires `transformers>=4.51.3`, which conflicts with the `transformers==4.49.0` that `spacy-transformers` expects. Pin `gliner==0.2.21` to avoid the conflict.

### LLM backend: choose one of two

The pipeline talks to an LLM through Ollama. You can use the Ollama cloud API or run Ollama locally.

**(a) Ollama cloud (API key)**

Create a `.env` file at the repo root with your key:

```
OLLAMA_API_KEY=<your-key>
```

When `OLLAMA_API_KEY` is set, the shared helper `ollama_utils.resolve_ollama_host()` auto-targets `https://ollama.com` and sends the key as `Authorization: Bearer <key>`. The optional `OLLAMA_HOST` env var can force a host (`cloud`, `localhost`, or a verbatim URL such as `http://localhost:1344`).

`.env` is gitignored. Never commit your key.

**(b) Local Ollama**

Install Ollama, start the server, and pull each model you want to use:

```
ollama serve
ollama pull gemma4:31b
ollama pull deepseek-v4-flash
ollama pull glm-5.1
```

Other models used in this repo: `deepseek-v4-pro`, `qwen3.5:397b`, `kimi-k2.7-code`, `mistral-large-3:675b`, `cogito-2.1:671b`. Without `OLLAMA_API_KEY` set, the pipeline auto-targets `localhost:11434`. Pass `--host localhost` (or set `OLLAMA_HOST=localhost`) to force local inference.

## Running the pipeline

Run the three stages in order. The commands below assume the Ollama cloud; add `--host localhost` to use a local Ollama server.

**1. NER**

```
python ner/ner.py \
  --model gemma4:31b \
  --mode fewshot \
  --language dutch \
  --temperature 0.0 \
  --input data/1816_third_letter.txt \
  --ollama-host cloud
```

Outputs a `.spacy` DocBin plus an offset map and meta JSON under `ner/ner-output/<model>/1816/`.

**2. Entity Linking**

```
python entity_linking/el.py \
  --model gemma4:31b \
  --temperature 0.0 \
  --top-k 3 \
  --input ner/ner-output/gemma4:31b/1816/1816_third_letter__gemma4-31b_t0.0_fewshot_dutch.spacy \
  --host cloud
```

If `--input` is omitted, an interactive picker lists the available `.spacy` files. Output goes to `entity_linking/el-results/`.

**3. Relation Extraction**

```
python relation_extraction/rel_extraction.py \
  --model gemma4:31b \
  --temperature 0.0 \
  --input entity_linking/el-results/<your>_el.spacy \
  --host cloud
```

If `--input` is omitted, an interactive picker lists `*_el.spacy` files. Output (events JSON, mention map, and `.ttl` RDF) goes to `output/re/` and `output/rdf/`.

**Evaluate each stage**

```
python ner/ner_evaluate.py -s ner/ner-output/<model>/1816/<file>.spacy
python entity_linking/el_evaluate.py --el entity_linking/el-results/<file>_el.spacy --gold entity_linking/1816_el_gs.csv
python relation_extraction/re_evaluate.py --events output/re/<file>_events.json --rdf output/rdf/<file>_events.ttl
```

### CLI flags

`ner.py`

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `gemma4:31b` | Ollama model name |
| `--mode`, `-M` | `fewshot` | Prompt strategy: `fewshot` or `zeroshot` |
| `--ollama-host`, `-H` | `cloud` | Ollama host: `cloud` or `localhost` |
| `--input`, `-i` | `data/1816_third_letter.txt` | Source `.txt` file relative to project root |
| `--split-mode`, `-S` | `brackets` | How to split text into pages: `brackets` (uses `[N]` markers) or `horizontal-rule` |
| `--temperature`, `-t` | `0.0` | LLM temperature (float) |
| `--language`, `-L` | `dutch` | Prompt language: `dutch` or `english` |

`el.py`

| Flag | Default | Description |
|---|---|---|
| `--model` | `gemma4:31b` | LLM model name |
| `--temperature` | `0.0` | LLM temperature |
| `--top-k` | `3` | Top-k rerank candidates |
| `--think` | model default | Thinking mode: `false`, `low`, `medium`, `high`. Use `false`/`low` for thinking models that over-think simple tasks |
| `--input` | interactive picker | Path to input `.spacy` file |
| `--host` | auto-switch | `cloud`, `localhost`, or a verbatim URL. Cloud when `OLLAMA_API_KEY` is set, else `localhost:11434` |

`rel_extraction.py`

| Flag | Default | Description |
|---|---|---|
| `--model` | constant | Ollama model (overrides the `OLLAMA_MODEL` constant) |
| `--temperature` | constant | Temperature (overrides the `TEMPERATURE` constant) |
| `--think` | model default | `true`, `false`, `low`, `medium`, `high` |
| `--host` | auto-switch | `cloud`, `localhost`, or a verbatim URL (same logic as `el.py`) |
| `--input` | interactive picker | Path to the input `*_el.spacy` file |
| `--offset-map` | auto-discovered | Path to the matching `offset_map.json` |
| `--regenerate-rdf` | off | Re-serialize `.ttl` from an existing `_events.json` + `_mention_map.json` without calling the LLM |
| `--events` | none | Path to an existing `*_events.json` (for `--regenerate-rdf`) |
| `--mention-map` | auto-discovered | Path to the matching `*_mention_map.json` (for `--regenerate-rdf`) |

## The `data/` folder

Holds the source material for the whole pipeline:

- `1816_third_letter.txt`, `1809_sixth_letter.txt`: manual transcriptions of the source letters (the 1816 letter uses inline `[N]` page markers).
- `GT_1816_for_mapping.txt`: full ground-truth 1816 transcription used by the HTR-correction step.
- `sample_annotations_brief_4_and_5.txt`: Recogito annotation notes for letters 4 and 5.
- `1816-scannumber-to-pagenumber.csv`: maps sequential scan numbers to printed page numbers.
- `page/`: 399 `.png` page scans plus 399 PAGE-XML files from Loghi HTR (`0552_0179_0001` through `0552_0179_0399`). Consumed by `ground_truth_mapping/` and `output/alto_exporter.py`.

See [`data/README.md`](data/README.md) for the scan-to-page mapping CSV columns.

## The `output/` folder

Holds the final scholarly-edition deliverables, split into subfolders:

- `alto/`: per-page coordinate-bearing ALTO-XML viewer edition (one `.alto.xml` plus `.overlay.html` per page), with page-scan images, `line_alignment.json`, and a shared `1816_third_letter.ttl`.
- `page/`: enriched PageXML files with NER+EL annotations written back into `<Metadata>`.
- `rdf/`: per-model `_events.ttl` RDF plus `ato_schacl_shapes.ttl` (SHACL shapes).
- `re/`: per-model RE output (`_events.json`, `_events.meta.json`, `_events.think.txt`, `_mention_map.json`).
- `tei/`: `1816_third_letter.tei.xml`, the canonical zero-loss TEI edition.

The exporter scripts sit at the `output/` root: `alto_exporter.py`, `tei_exporter.py`, `pagexml_enricher.py`, `alto_overlay.py`, `alto_selfcheck.py`, `tei_selfcheck.py`, `ato_schacl_rdf_validator.py`. See [`output/README.md`](output/README.md).

## Repo layout

```
hist-dutch-travelogues-nlp/
├── data/                  # source texts, scan images, PAGE-XML (README)
├── ner/                   # Stage 1: NER (README)
├── entity_linking/        # Stage 2: EL to Wikidata/GeoNames (README)
├── relation_extraction/   # Stage 3: RE + ATO RDF (README)
├── output/                # Stage 4: ALTO/TEI/RDF export (README)
├── GliNER_baseline/       # non-LLM NER baseline (README)
├── ground_truth_mapping/  # HTR correction of PAGE-XML (README)
├── SLURM/                 # HPC batch submission (README)
├── ollama_utils.py        # shared Ollama streaming + host-resolution helper
└── requirements.txt
```

> _Disclaimer_: AI (LLMs) tools were leveraged to optimize development and ensure code robustness.
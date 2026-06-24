# hist-dutch-travelogues-nlp

An integrated NLP pipeline for Named Entity Recognition (NER), Entity Linking (EL), Relation Extraction (RE), and RDF generation for 19th-century Dutch travelogues. Part of the **Academic Tourism project** at the **University of Groningen**.

The primary source material is letters from a Groningen student traveling through Germany (~1816) as part of his *Bildungsreise* (academic grand tour). The pipeline extracts travel events according to the **Academic Tourism Ontology (ATO)**, an extension of **CIDOC CRM** for modelling academic tourism phenomena.

The pipeline is implemented in Python using spaCy for NER, Ollama-hosted LLMs for all NLP stages, and rdflib for RDF generation and validation.

---

## Table of Contents

- [Pipeline Architecture](#pipeline-architecture)
- [Requirements and installation](#requirements-and-installation)
- [Environment variables](#environment-variables)
- [Data files](#data-files)
- [Stage 1: Named Entity Recognition (NER)](#stage-1-named-entity-recognition-ner)
- [Stage 2: Entity Linking (EL)](#stage-2-entity-linking-el)
- [Stage 3: Relation Extraction and RDF Generation (RE)](#stage-3-relation-extraction-and-rdf-generation-re)
- [Stage 4: Output and Export](#stage-4-output-and-export)
- [Auxiliary: Ground Truth-to-PageXML Mapping](#auxiliary-ground-truth-to-pagexml-mapping)
- [Evaluation](#evaluation)
- [Model configuration reference](#model-configuration-reference)
- [Interactive file selection pattern](#interactive-file-selection-pattern)
- [Important architectural notes](#important-architectural-notes)
- [File and directory reference](#file-and-directory-reference)
- [End-to-end running guide](#end-to-end-running-guide)

---

## Pipeline Architecture

The pipeline consists of four sequential stages, each depending on the output of the previous one, plus an auxiliary module for ground truth correction of HTR output.

```
                    ┌─────────────────────────────────┐
                    │  Source text / PageXML           │
                    │  data/*.txt  |  data/page/*.xml  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  STAGE 1: NER                    │
                    │  ner/ner.py                      │
                    │  spaCy + spacy-llm + Ollama      │
                    │  8 CIDOC CRM entity labels       │
                    │  ───────────────────────────     │
                    │  Output: .spacy DocBin           │
                    │          *offset_map.json        │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  STAGE 2: Entity Linking (EL)    │
                    │  entity_linking/el.py            │
                    │  3-stage LELA-inspired:          │
                    │    Candidate Generation          │
                    │    → Reranking                    │
                    │    → Selection                    │
                    │  Knowledge bases: Wikidata +     │
                    │  GeoNames                         │
                    │  ───────────────────────────     │
                    │  Output: *_el.spacy              │
                    │          (kb_id_ set on entities) │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  STAGE 3: Relation Extraction    │
                    │  + RDF Generation                 │
                    │  relation_extraction/            │
                    │    rel_extraction.py              │
                    │  Extracts travel events (ATO/    │
                    │  CIDOC CRM), generates RDF/TTL   │
                    │  ───────────────────────────     │
                    │  Output: *_events.json           │
                    │          *_mention_map.json       │
                    │          *_events.ttl (RDF)      │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  STAGE 4: Output & Export        │
                    │  output/                          │
                    │    tei_exporter.py               │
                    │    pagexml_enricher.py (WIP)     │
                    │    ato_schacl_rdf_validator.py   │
                    │  ───────────────────────────     │
                    │  Output: *.tei.xml               │
                    │          enriched PageXML        │
                    │          SHACL validation report │
                    └─────────────────────────────────┘

    AUXILIARY:
    ┌─────────────────────────┐     ┌──────────────────────────────┐
    │ GT-to-PageXML Mapping   │     │ Evaluation:                  │
    │ ground_truth_mapping/   │     │ ner_evaluate.py              │
    │ Corrects noisy HTR      │     │ el_evaluate.py               │
    │ via Ollama GT alignment │     │                              │
    └─────────────────────────┘     └──────────────────────────────┘
```

---

## Requirements and installation

**Python**: 3.10+

**Core dependencies:**

| Package | Purpose |
|---|---|
| `spacy` | NLP framework |
| `spacy-llm` | LLM integration for spaCy pipelines |
| `ollama` | Python client for Ollama API |
| `httpx` | HTTP client (for Ollama streaming) |
| `rdflib` | RDF/Turtle manipulation (`pip install rdflib`) |
| `pyshacl` | SHACL validation (optional, Stage 4 only) |
| `jinja2` | Prompt templating |
| `pyyaml` | Loading few-shot examples |
| `pandas` | CSV handling, evaluation reports |
| `rapidfuzz` | Fuzzy string matching (GeoNames gazetteer) |
| `requests` | Wikidata / Wikipedia API calls |
| `python-dotenv` | Environment variable loading from `.env` |

**Installation:**

```bash
pip install -r requirements.txt
```

No pre-trained spaCy models are required. The pipeline uses blank (`lang = "nl"`) models throughout.

---

## Environment variable

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OLLAMA_API_KEY` | Yes, for cloud mode | — | API key for Ollama cloud service at `https://ollama.com`. Required when `--ollama-host cloud` (or `OLLAMA_HOST = "https://ollama.com"`). |


The `.env` file at the project root is loaded via `python-dotenv` in the main scripts. It is gitignored.

---

## Data files

| Path | Type | Description |
|---|---|---|
| `data/1816_third_letter.txt` | Text | Source travelogue (~1816, third letter), ~21 KB. Pages split with `[N]` bracket markers. |
| `data/1809_sixth_letter.txt` | Text | Earlier letter (~1809), ~23 KB. Pages separated by horizontal rules (lines of underscores). |
| `data/sample_annotations_brief_4_and_5.txt` | Text | Sample annotations for letters 4 and 5, ~36 KB. |
| `data/GT_1816_for_mapping.txt` | Text | Full ground truth transcription of the 1816 letter with `[Page N]` markers, ~464 KB. Used by GT-to-PageXML mapping. |
| `data/1816-scannumber-to-pagenumber.csv` | CSV | Maps digitization scan numbers (integer) to manuscript page numbers (Roman or Arabic string). Columns: `Scan Number`, `Page Number`. |
| `data/page/*.xml` | PageXML | HTR (Handwritten Text Recognition) output in PageXML format (PAGE 2013-07-15 namespace). Paired with corresponding `*.png` page images. |
| `data/page/*.png` | Image | Page scan images. |
| `data/page_updated/` | PageXML | Corrected PageXML files produced by the GT-to-PageXML mapping module. |
| `ner/gs_annotations/1816.jsonld` | JSON-LD | Recogito manual NER annotations (~327 KB). Contains W3C Web Annotation format annotations (see format description below). |
| `entity_linking/GeoNames_DE_gazetteer.txt` | TSV | Local GeoNames gazetteer for Germany (~28 MB). Loaded at runtime by the EL candidate generation module. |

### Recogito Annotation Format

The ground truth annotations at `ner/gs_annotations/1816.jsonld` follow the [W3C Web Annotation](https://www.w3.org/TR/annotation-model/) format as exported by [Recogito](https://recogito.pelagios.org/). Each annotation (`type: Annotation`) contains:

- **Body** — an array of `TextualBody` objects:
  - A **tagging body** with `purpose: "tagging"` whose `id` is a CIDOC CRM type URI (e.g., `http://www.cidoc-crm.org/cidoc-crm/E53`) and `value` is the label (e.g., `"E53 Place"`).
  - An **identifying body** with `purpose: "identifying"` and `value: "PLACE"` (or `"THING"`).
  - An optional **commenting body** with `purpose: "commenting"` containing URLs to Wikidata, GeoNames, and Meyers Gazetteer references (e.g., `"geonames.org/2914314/\nmeyersgaz.org/place/10671043\nhttps://www.wikidata.org/wiki/Q1550705"`).
- **Target** — an object with selectors:
  - `TextPositionSelector` — character offsets (`start`, `end`)
  - `TextQuoteSelector` — exact text match (`exact`)

The CIDOC-to-pipeline label mapping used for evaluation is:

| CIDOC URI | Pipeline label |
|---|---|
| `E53` | `E53_Place` |
| `E18` | `E18_Physical_Thing` |
| `E22` | `Mode_of_Transportation` |

---

## Stage 1: Named Entity Recognition (NER)

The NER module uses spaCy's `spacy-llm` integration to annotate the source text with entities. It supports two text input modes and two prompting languages.

### Entity labels (8)

| Label | CIDOC / LRMoo type | Description |
|---|---|---|
| `E53_Place` | E53 Place | Geopolitical entities (cities, towns, regions), geographic areas, and rooms/spaces within buildings. |
| `E18_Physical_Thing` | E18 Physical Thing | Tangible physical structures: buildings, bridges, monuments, mountains. |
| `Mode_of_Transportation` | — (renamed from E22) | Transport modes: active (walking) and vehicle-based (carriage, coach). |
| `E52_Time_Span` | E52 Time-Span | Temporal expressions: dates, times, durations, time intervals. |
| `F2_Expression` | F2 Expression (LRMoo) | Artistic/intellectual creations (paintings, sculptures, compositions). The immaterial expression, not the physical carrier. |
| `E19_Physical_Object` | E19 Physical Object | Movable non-biological natural objects (fossils, minerals, shells). |
| `E20_Biological_Object` | E20 Biological Object | Movable biological specimens (eggs, taxidermy, skeletons). Former living things. |
| `E31_Document` | E31 Document | Specific identifiable documents, books, or publications. |

### Script: `ner/ner.py`

Main NER pipeline. Loads source text, splits into pages, runs NER on each page via spacy-llm, and saves merged results.

**CLI arguments:**

| Argument | Flag | Default | Choices | Description |
|---|---|---|---|---|
| `--model` | `-m` | `gemma4:31b` | Any Ollama model | LLM model for NER inference |
| `--mode` | `-M` | `fewshot` | `zeroshot`, `fewshot` | Prompt strategy: few-shot with examples or zero-shot |
| `--ollama-host` | `-H` | `cloud` | `cloud`, `localhost` | API host (`cloud` uses `https://ollama.com`, requires `OLLAMA_API_KEY`) |
| `--input` | `-i` | `data/1816_third_letter.txt` | Path | Source text file (relative to project root) |
| `--split-mode` | `-S` | `brackets` | `brackets`, `horizontal-rule` | Text splitting strategy: `brackets` splits on `[N]` markers, `horizontal-rule` splits on lines of 16+ underscores |
| `--temperature` | `-t` | `0.0` | Float | LLM sampling temperature |
| `--language` | `-L` | `dutch` | `dutch`, `english` | Prompt language |

**Hardcoded configuration** (not exposed as CLI arguments):

| Constant | Value | Description |
|---|---|---|
| `INPUT_MODE` | `"txt"` | Source type: `"txt"` reads from `--input`, `"pagexml"` reads corrected PageXML from `data/page_updated/` (requires code edit to change) |
| `PAGEXML_DIR` | `data/page_updated/` | PageXML directory (used when `INPUT_MODE = "pagexml"`) |
| `SCAN_CSV` | `data/1816-scannumber-to-pagenumber.csv` | Scan-to-page mapping (used in PageXML mode) |
| `OUTPUT_DIR` | `ner/ner-output/` | Base output directory. Files are written to `ner-output/{model}/{letter_label}/` subdirectories. |

**Retry logic:** up to 3 retries per page if the LLM returns an API error or produces zero entities. Failed pages are tracked in the metadata JSON.

**Input file:**

- Source `.txt` file with page markers (when `INPUT_MODE = "txt"`).
- spaCy LLM config file selected based on language and mode (see [Config file mapping](#config-file-mapping) below).

**Output files** (to `ner/ner-output/{model}/{letter_label}/`):

| File pattern | Description |
|---|---|
| `{letter}__{model}_t{temp}_{mode}_{lang}.spacy` | Merged spaCy DocBin with entity annotations, pages in order |
| `{letter}__{model}_t{temp}_{mode}_{lang}_offset_map.json` | Dictionary mapping `{page_number: character_offset_in_full_text}` |
| `{letter}__{model}_t{temp}_{mode}_{lang}_line_map.json` | (PageXML mode only) Page-local character offset to HTR line index |
| `{letter}__{model}_t{temp}_{mode}_{lang}_meta.json` | Execution metadata (model, mode, temperature, timing, retry pages) |

### NER configuration files (`ner/config/`)

The NER config files are spaCy `spacy-llm` configuration files that define labels, prompt templates, few-shot examples, and model settings.

**Config file mapping** (selected by `--language` and `--mode`):

| Language | Mode | Config file |
|---|---|---|
| `dutch` | `fewshot` | `ner/config/fewshot.cfg` |
| `dutch` | `zeroshot` | `ner/config/zeroshot.cfg` |
| `english` | `fewshot` | `ner/config/fewshot_english.cfg` |
| `english` | `zeroshot` | `ner/config/zeroshot_english.cfg` |

**Template files:**

| File | Description |
|---|---|
| `ner/config/prompt_dutch.jinja` | Jinja2 NER prompt template (Dutch). Instructs the LLM on entity definitions, annotation rules, and output format. |
| `ner/config/prompt_english.jinja` | Jinja2 NER prompt template (English). |
| `ner/config/fewshot_examples.yaml` | 13 few-shot examples spanning all 8 entity labels. Each example contains annotated source text with entity spans. |

**Model settings in config files** (overridable via CLI `--model`):

| Setting | Value |
|---|---|
| Model | `deepseek-v4-pro` |
| API URL | `https://ollama.com` |
| Context length | 256 000 tokens |
| Temperature (default) | 0.1 |

### Script: `ner/ner_evaluate.py`

Evaluates NER predictions against the Recogito ground truth.

**CLI arguments:**

| Argument | Flag | Description |
|---|---|---|
| `--spacy` / `-s` | Optional | Path to predicted `.spacy` file. If omitted, presents an interactive file picker from `ner/ner-output/`. |

**Input files:**

- Predicted `.spacy` DocBin from `ner/ner-output/`
- Matching `*offset_map.json` from `ner/ner-output/`
- Recogito ground truth at `ner/gs_annotations/{prefix}.jsonld` (prefix derived from the spacy filename)

**Output files** (to `ner/ner-evaluation/`):

| File | Description |
|---|---|
| `scores.csv` | Cumulative per-label and overall scores, appended on each run |
| `{spacy_stem}_errors.csv` | Instance-level errors with gold/predicted spans |

**Methodology:**

- **Relaxed matching**: any overlap between predicted and gold span of the same label counts as a match.
- **Strict matching**: exact character-offset match (via spaCy `Scorer`).
- Per-label Precision, Recall, and F1 are reported for both strict and relaxed scoring.

### Script: `ner/ner_displacy.py`

Launches a browser-based visualization of NER/EL results using spaCy displaCy.

- Interactively selects any `.spacy` file from `ner/ner-output/` or `entity_linking/el-results/`.
- When EL results are loaded, attaches Wikidata KB IDs as hyperlinks.
- Colour-coded entity labels: E53_Place=blue, E18_Physical_Thing=green, Mode_of_Transportation=yellow, E52_Time_Span=red, F2_Expression=purple, E19_Physical_Object=light green, E20_Biological_Object=darker green, E31_Document=orange.

### Script: `ner/export_gs_to_el_template.py`

Creates an Entity Linking gold standard CSV template from the Recogito ground truth.

- Reads `ner/gs_annotations/1816.jsonld` (hardcoded path).
- Extracts every annotated entity with CIDOC type, character offsets, and — where available — pre-fills Wikidata QID and GeoNames ID from the Recogito comment bodies.
- Output: `ner/gs_el_gold_template.csv` with columns: `text`, `label`, `start_char`, `end_char`, `wikidata_qid`, `geonames_id`, `note`.

### Script: `ner/export_ner_to_el_template.py`

Creates an Entity Linking gold standard CSV template from NER pipeline output.

- Interactively selects a `.spacy` file from `ner/ner-output/`.
- Extracts every entity with page number, text, label, and character offsets.
- Output: `ner/ner-output/{stem}_el_gold_template.csv` with empty `wikidata_qid` and `geonames_id` columns for manual annotation.
- Also accepts file path as CLI argument `sys.argv[1]`.

### Batch runner scripts

| Script | Description |
|---|---|
| `ner/ner.sh` | Runs NER for a single model configuration (fewshot, gemma4:31b, cloud). Saves log and spacy output. |
| `ner/ner_and_eval.sh` | Runs NER + evaluation for multiple models. Iterates over a model list, performs NER and then evaluation for each. Supports both cloud and localhost Ollama. |

---

## Stage 2: Entity Linking (EL)

The EL module uses a three-stage LELA-inspired (Learning-based Entity Linking Architecture) pipeline to link NER entities to Wikidata Q-IDs and GeoNames IDs. Each stage uses the full page as context.

### Architecture

```
For each entity in the spaCy Doc:

  ┌─────────────────────────────────────────────┐
  │ Stage 1: Candidate Generation                │
  │ el_candidates.py                             │
  │                                              │
  │ Sparse lane: Wikidata text search (nl/en/de) │
  │  surface form                               │
  │                                              │
  │ Dense/EPG lane:                              │
  │  LLM predicts modern profile from archaic     │
  │  context (modern_name, region, type) →       │
  │  Wikidata SPARQL hybrid search →              │
  │  Wikipedia full-text →                        │
  │  GeoNames gazetteer (rapidfuzz matching)     │
  │                                              │
  │ Output: candidate lists                      │
  │  {wikidata: [...], geonames: [...]}          │
  └──────────────────┬──────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────┐
  │ Stage 2: Reranking                           │
  │ el_reranker.py                               │
  │                                              │
  │ LLM pointwise scoring (0-10) of each          │
  │ candidate against entity + page context       │
  │                                              │
  │ Output: top-k candidates with rerank_score   │
  └──────────────────┬──────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────┐
  │ Stage 3: Selection                           │
  │ el_selector.py                               │
  │                                              │
  │ Wikipedia extract enrichment for top-k        │
  │                                              │
  │ LLM selects best candidate or returns NIL     │
  │                                              │
  │ Output: KB ID (QID, gn:NNN, or NIL)          │
  └─────────────────────────────────────────────┘
```

**Entity types skipped from EL** (get `None` KB IDs):

- `Mode_of_Transportation`
- `E52_Time_Span`
- `F2_Expression`
- `E19_Physical_Object`
- `E20_Biological_Object`
- `E31_Document`

These non-geographical entity types do not correspond to Wikidata items with meaningful locations.

### Script: `entity_linking/el.py`

Orchestrator that runs all three EL stages.

**Configuration (hardcoded):**

| Constant | Value |
|---|---|
| `MODEL_NAME` | `gemma4:31b-cloud` |
| `TOP_K_RERANK` | 3 |
| Ollama URL | `https://ollama.com` (with `OLLAMA_API_KEY`) or `http://localhost:11434` |

**Input:** Interactively selects `.spacy` from `ner/ner-output/`.

**Output files** (to `entity_linking/el-results/`):

| File | Description |
|---|---|
| `{stem}_el.spacy` | DocBin with `kb_id_wikidata_` and `kb_id_geonames_` set on spans and tokens |
| `{stem}_offset_map.json` | Copy of the matching offset map from NER output |

**Custom spaCy extensions registered:**

- `Span._.kb_id_wikidata_` / `Token._.ent_kb_id_wikidata_`
- `Span._.kb_id_geonames_` / `Token._.ent_kb_id_geonames_`

### Script: `entity_linking/el_candidates.py` (Stage 1)

Candidate generation with two parallel lanes:

**Sparse lane:** Direct Wikidata EntitySearch API query for the surface form in Dutch, English, and German.

**Dense/EPG (Entity Profile Generation) lane:** The LLM predicts a modern entity profile (JSON: `modern_name`, `country_or_region`, `type_keywords`) from the archaic context, then queries:
- **Wikidata SPARQL**: EntitySearch with FILTER on location terms from the LLM profile.
- **Wikipedia full-text search**: English Wikipedia, resolved to Wikidata Q-IDs.
- **GeoNames gazetteer**: Local file `entity_linking/GeoNames_DE_gazetteer.txt` (~28 MB, tab-separated) matched via rapidfuzz.

**Caching:** Results are cached in `_cache` (keyed by `entity_text|entity_label`) and `_llm_cache` (keyed by `entity_text|entity_label|context[:300]`). Call `clear_cache()` to reset.

**User-Agent:** `DutchTravelogueNLP/1.0 (https://github.com/tijn-do/hist-dutch-travelogues-nlp; research project)`

### Script: `entity_linking/el_reranker.py` (Stage 2)

LLM pointwise scoring of all candidates. A Dutch-language prompt asks the LLM to rate each candidate 0-10 based on fit to entity text and page context.

**Temperature:** 0.0. No heuristic fallback — errors propagate as `ERROR:*` markers.

**Output:** Top-k candidates sorted by `rerank_score` (descending).

### Script: `entity_linking/el_selector.py` (Stage 3)

Candidate selection with Wikipedia enrichment.

- **Enrichment:** Resolves top-k Wikidata Q-IDs to English Wikipedia titles via `wbgetentities`, then fetches page summaries via Wikipedia REST API (`/page/summary/{title}`, capped at ~300 words).
- **Selection:** An LLM prompt with entity text, page context, and enriched candidate info asks for the best match.
- **Output:** Selected ID (Wikidata Q-ID, `gn:{geonameid}`, or `"NIL"`).

**Temperature:** 0.0. Even single candidates are verified.

### Script: `entity_linking/el_stats.py`

Displays linking statistics for a saved `_el.spacy` file.

- Interactively selects from `entity_linking/el-results/`.
- Reports: total entities, Wikidata-linked count, GeoNames-linked count, both, neither, and error counts.

### Script: `entity_linking/el_evaluate.py`

Evaluates EL predictions against a gold standard CSV.

**CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--gold` | `ner/gs_el_gold_template.csv` | Gold CSV with `wikidata_qid` and `geonames_id` columns |
| `--el` | Interactive selection | Path to `_el.spacy` file |
| `--detail` | `True` | Write per-instance detail CSV |

**Methodology:**

- **Relaxed span matching**: overlap-based alignment between gold and predicted entities.
- **Two levels of reporting:**
  1. **EL-only**: conditioned on predicted entities that overlap a gold span (isolates linking accuracy from NER detection errors).
  2. **Full (end-to-end)**: includes entities NER missed entirely.
- Separate metrics for Wikidata QID, GeoNames ID, and combined.
- Output: `{el_path.stem}_el_eval.csv` with per-instance TP/FP/FN records.

---

## Stage 3: Relation Extraction and RDF Generation (RE)

The RE module loads NER+EL output (`_el.spacy`), builds annotated text with inline entity mention IDs, queries an LLM to extract travel events according to the Academic Tourism Ontology (ATO), and generates RDF/Turtle output.

### Event types

| Event type | CIDOC CRM class | Description |
|---|---|---|
| `translocation` | E9_Move | Movement between places with origin and destination |
| `indoor_tour` | E7_Activity | Visit inside a building with rooms/sections as sub-events |
| `outdoor_tour` | E7_Activity | Walk through a park, garden, or city |
| `stay` | E7_Activity | Overnight accommodation, meal, or rest stop |
| `mentioned_only_places` | — | Places mentioned in the text but not visited |

### Script: `relation_extraction/rel_extraction.py`

**Configuration (hardcoded):**

| Constant | Value |
|---|---|
| `OLLAMA_MODEL` | `deepseek-v4-flash` |
| `OLLAMA_HOST` | `https://ollama.com` |
| `TEMPERATURE` | 0.0 |
| `EXAMPLES_FILE` | `relation_extraction/re_examples.yaml` |
| `PROMPT_TEMPLATE` | `relation_extraction/re_prompt_dutch.jinja` |
| `OUTPUT_JSON` | `output/re/1816_third_letter_events.json` |
| `OUTPUT_RDF` | `output/rdf/1816_third_letter_events.ttl` |

**Pipeline steps:**

1. **Select input**: Interactively picks `_el.spacy` from `entity_linking/el-results/`.
2. **Build annotated text**: Strips `[N]` page markers and builds full text with inline entity mention markers: `[Entity Name](LABEL, eN)` (where `eN` is a unique mention ID). Generates a `mention_map` for ID-to-entity resolution.
3. **Load few-shot examples** from `re_examples.yaml`.
4. **Build prompt** using the Jinja2 template `re_prompt_dutch.jinja` (capped at 200,000 characters to stay within the model's context window).
5. **Query LLM** via `ollama_utils.stream_ollama_chat()` with 600-second timeout.
6. **Parse JSON** from the LLM response (extracts from markdown code fences if present).
7. **Validate** all mention IDs against the mention_map.
8. **Save** events JSON and mention_map JSON.
9. **Generate RDF** in Turtle format using `rdflib`.

**RDF namespaces used:**

| Prefix | Namespace URI |
|---|---|
| `cidoc-crm` | `http://www.cidoc-crm.org/cidoc-crm/` |
| `ato` | `http://academictourism.com/entity/` |
| `academictourism` | `http://academictourism.com/academictourism#` |
| `skos` | `http://www.w3.org/2004/02/skos/core#` |
| `lrmoo` | `http://iflastandards.info/ns/lrm/lrmoo/` |

**RDF generation patterns:**

| Pipeline label | CIDOC type | Entity URI prefix |
|---|---|---|
| `E53_Place` | E53 Place | `LOC.*` |
| `E18_Physical_Thing` | E18 Physical Thing | `LOC.*` |
| `Mode_of_Transportation` | E55 Type / ato:Mode_of_Transportation | `MOD.*` |
| `E52_Time_Span` | E52 Time-Span | `TS.*` |
| `F2_Expression` | F2 Expression (LRMoo) | `ART.*` |
| `E19_Physical_Object` | E19 Physical Object | `PHO.*` |
| `E20_Biological_Object` | E20 Biological Object | `BIO.*` |
| `E31_Document` | E31 Document | `DOC.*` |

Key RDF patterns include:
- Events typed as both `E7_Activity` and `E92_Spacetime_Volume`.
- `P183_ends_before_the_start_of` for temporal ordering.
- `P10_contains` / `P10_falls_within` for event mereology (sub-events).
- `P26_moved_to` / `P27_moved_from` for translocations.
- `P7_took_place_at` for tours and stays.
- `P8_took_place_on_or_within` for indoor tours.
- CT.* citation entities as `E89_Propositional_Object` with `P129_is_about`, `P67_refers_to`, and `academictourism:conveys`.
- `skos:closeMatch` for Wikidata and GeoNames external links.

**Output files** (all three written simultaneously):

| File | Description |
|---|---|
| `output/re/{letter}_events.json` | Raw LLM output as structured JSON |
| `output/re/{letter}_mention_map.json` | Entity mention ID-to-span/label/kb_id mapping |
| `output/rdf/{letter}_events.ttl` | CIDOC CRM + ATO RDF/Turtle graph |

### Supporting files: `relation_extraction/re_prompt_dutch.jinja`

Jinja2 prompt template instructing the LLM on:
- Event type definitions (translocation, indoor tour, outdoor tour, stay).
- Sub-event rules (2+ rooms/sections each get their own sub-event).
- Time span handling.
- F2_Expression handling.
- Visited vs. Merely Mentioned distinction.
- JSON output format specification.

### Supporting files: `relation_extraction/re_examples.yaml`

Five few-shot examples covering:
- Translocation with via-points.
- Museum tour with sub-events.
- Park walk (outdoor tour).
- Time span examples.
- Mixed event sequences.

### Script: `relation_extraction/query_route.py`

Displays the chronological travel route by loading an events TTL file and building the P183 temporal chain.

**Usage:**

```bash
python relation_extraction/query_route.py [path/to/events.ttl] [--csv]
```

Defaults to an interactive file picker from `output/rdf/` for `*_events.ttl` files.

**Output modes:**
- **Default (human-readable table)**: Shows translocations, tours, and stays in chronological order with place labels, Wikidata IDs, CT.* citation provenance, and nested sub-events. Includes a `mentioned_only` places section and summary statistics.
- **`--csv`**: Structured CSV output with columns: `step`, `event_id`, `parent_id`, `type`, `label`, `from`/`from_qid`, `to`/`to_qid`, `at`/`at_qid`, `within`/`within_qid`, `mode`, `citations`.

### Script: `relation_extraction/validate_rdf.py`

Lightweight schema validation of generated RDF against ATO constraints (12 rules).

**Usage:**

```bash
python relation_extraction/validate_rdf.py [path/to/events.ttl]
```

Defaults to `output/rdf/1816_third_letter_events.ttl`.

**Validation rules:**

| # | Rule | Severity |
|---|---|---|
| 1 | Every event typed E7_Activity or E9_Move | Error |
| 2 | Every E9_Move (except journey root) must have P26_moved_to and P27_moved_from | Error |
| 3 | Every non-move E7_Activity should have P7_took_place_at | Warning |
| 4 | P7/P26/P27 targets typed E53_Place | Error |
| 5 | P8 targets typed E18/E19/E20 | Warning |
| 6 | P183 chain unbroken (one first, one last) | Error |
| 7 | No duplicate event IDs | Error |
| 8 | No place both mentioned (P67) and visited (P7/P26/P27) | Error |
| 9 | Journey root has P10_contains | Error |
| 10 | Events have P11_had_participant | Warning |
| 11 | Event ID convention (BRF prefix, RS pattern) | Error |
| 12 | Events with Time_Span citations have P4_has_time-span | Warning |
| 13 | Every P4_has_time-span target typed E52_Time-Span | Error |

---

## Stage 4: Output and Export

### Script: `output/tei_exporter.py`

Produces a self-contained TEI P5 XML document from NER+EL annotations and optionally from RE pipeline outputs.

**Input:** Interactively selects `_el.spacy` from `entity_linking/el-results/`. Auto-discovers the matching offset map and, if available, the RE mention map and events TTL.

**Entity-to-TEI mapping:**

| Pipeline label | TEI element | Attributes |
|---|---|---|
| `E53_Place` | `<placeName>` | `xml:id`, `type="E53_Place"`, `ref`, `key` |
| `E52_Time_Span` | `<date>` | `xml:id`, `type="E52_Time_Span"` |
| `F2_Expression` | `<rs>` | `xml:id`, `type="F2_Expression"` |
| Other labels | `<rs>` | `xml:id`, `type="..."` |

**Features:**
- `<pb n="page"/>` page milestones for each source page.
- `<lb/>` linebreak milestones at each newline.
- `xml:id` attributes matching RE mention IDs (e.g. `e5` ↔ `CT.BRF0003.e5`) for bidirectional linking between TEI and RDF.
- `ref` (Wikidata QID) and `key` (GeoNames ID) attributes for KB linking.
- `<standOff>` with embedded RDF/XML containing ATO triples (when RE pipeline output is available).

**Output:** `output/tei/1816_third_letter.tei.xml` (hardcoded path).

### Script: `output/pagexml_enricher.py`

**WORK IN PROGRESS.** Writes NER + EL annotations back into PageXML `<Metadata>` sections.

**Inputs:**
- Hardcoded paths to `_el.spacy`, offset map, and line map.
- Corrected PageXML from `data/page_updated/` (falls back to `data/page/` if empty).

**Output:** Enriched PageXML files in `output/page/` with `<MetadataItem type="ner-annotations">` containing per-entity `<Annotation>` elements with: `label`, `start`, `end`, `line`, `kb_id`, `kb_id_wikidata`, `kb_id_geonames`.

### Script: `output/ato_schacl_rdf_validator.py`

Validates generated events RDF against the ATO SHACL shapes graph.

**Usage:**

```bash
python output/ato_schacl_rdf_validator.py [path/to/events.ttl]
```

Defaults to interactive file picker from `output/rdf/`.

**Input files:**

| File | Description |
|---|---|
| Events TTL | Generated by `relation_extraction/rel_extraction.py` |
| `output/rdf/ato_schacl_shapes.ttl` | SHACL shapes (~902 KB, ~623 shapes) |
| `output/rdf/ATO.rdf` | ATO ontology (~1 MB) — loaded for class/property definitions |

**Requires:** `pyshacl` (`pip install pyshacl`).

The validator repairs incomplete PropertyShapes (missing `sh:path`) before validation.

---

## Auxiliary: Ground Truth-to-PageXML Mapping

Corrects noisy HTR (Handwritten Text Recognition) output in PageXML files by aligning each HTR line to its correct ground truth transcription using an Ollama LLM.

### Script: `ground_truth_mapping/gt_to_pagexml_local.py`

**Configuration (hardcoded):**

| Constant | Value |
|---|---|
| `GT_FILE` | `data/GT_1816_for_mapping.txt` |
| `PAGE_DIR` | `data/page/` |
| `OUTPUT_DIR` | `data/page_updated/` |
| `MODEL_NAME` | `gemma4:31b-cloud` |
| `TEMPERATURE` | 0.0 |

**Pipeline per PageXML file:**
1. Extract scan number from the XML filename.
2. Map scan number to page number via `data/1816-scannumber-to-pagenumber.csv`.
3. Retrieve ground truth transcription for that page from `GT_1816_for_mapping.txt`.
4. Align HTR lines to ground truth via an LLM call (Ollama with `format="json"`).
5. Produce corrected PageXML in `data/page_updated/`.

Empty pages and pages without a mapping or ground truth are copied as-is.

**Output files:**

| File | Description |
|---|---|
| `data/page_updated/*.xml` | Corrected PageXML files |
| `data/page_updated/report_summary.csv` | Per-page mapping statistics (total_lines, corrected_lines, unmatched_lines, confidence, fuzzy_accuracy, gt_coverage_pct, status) |
| `data/page_updated/report_unmatched.csv` | Detail of HTR lines that could not be mapped (filename, line_index, original_text) |

**`report_summary.csv` columns:**

| Column | Type | Description |
|---|---|---|
| `filename` | str | PageXML filename |
| `total_lines` | int | Total HTR text lines on the page |
| `corrected_lines` | int | Lines successfully mapped to ground truth |
| `unmatched_lines` | int | Lines with no GT equivalent |
| `confidence` | float | LLM-reported alignment confidence (0.0–1.0) |
| `fuzzy_accuracy` | float | SequenceMatcher similarity: recovered vs. original GT (0–100%) |
| `gt_chars_total` | int | Total characters in the ground truth page |
| `gt_chars_mapped` | int | Characters covered by the mapping |
| `gt_coverage_pct` | float | Percentage of GT characters covered |
| `status` | str | `mapping_attempted`, `no_mapping`, or `no_gt` |

### Script: `ground_truth_mapping/gt_to_pagexml_hpc.py`

Equivalent to `gt_to_pagexml_local.py` but adapted for the Habrok HPC cluster:
- Model: `gemma4:31b` (no `-cloud` suffix).
- Ollama timeout: 1200 seconds (vs. 600 seconds for local).
- No API key — Ollama runs locally on the compute node.

### SLURM batch job: `SLURM/gt_to_pagexml_mapping.sh`

**SBATCH configuration:**

| Setting | Value |
|---|---|
| Partition | `gpu` |
| Nodes | 1 |
| CPUs | 1 |
| Memory | 32 GB |
| Time limit | 3 hours |
| GPU | 1 |

**Execution:**
1. Sets `OLLAMA_MODELS` to `/scratch/$USER/ollama_models_scratch`.
2. Starts an Ollama server on the compute node.
3. Pulls `gemma4:31b` and loads it into GPU memory.
4. Activates the Python virtual environment at `$HOME/venvs/ollama/bin/activate`.
5. Runs `gt_to_pagexml_hpc.py`.

---

## Evaluation

### NER evaluation

| Component | Detail |
|---|---|
| Script | `ner/ner_evaluate.py` |
| Gold standard | `ner/gs_annotations/{prefix}.jsonld` (Recogito W3C JSON-LD) |
| Metrics | Precision, Recall, F1 per label (strict and relaxed) |
| Output | `ner/ner-evaluation/scores.csv` (cumulative) + `*_errors.csv` (per-run) |
| Gold generation | `ner/export_gs_to_el_template.py`, `ner/export_ner_to_el_template.py` |

### EL evaluation

| Component | Detail |
|---|---|
| Script | `entity_linking/el_evaluate.py` |
| Gold standard | `ner/gs_el_gold_template.csv` (auto-generated from Recogito, manually curated) |
| Metrics | Wikidata P/R/F1, GeoNames P/R/F1, combined P/R/F1 |
| Levels | EL-only (conditioned on span overlap) and Full (end-to-end, includes NER misses) |
| Output | `*_el_eval.csv` (per-instance TP/FP/FN) |

### Gold standard generation workflow

1. `ner/export_gs_to_el_template.py` — Reads Recogito annotations, creates CSV with pre-filled Wikidata/GeoNames IDs from the annotation comment bodies.
2. `ner/export_ner_to_el_template.py` — Reads NER output, creates CSV with empty ID columns for manual annotation.
3. Manual curation: fill in missing Wikidata QID and GeoNames ID values.
4. `entity_linking/el_evaluate.py` — Compare EL predictions against the curated gold standard.

---

## Model configuration reference

| Model | Provider | Used in | Notes |
|---|---|---|---|
| `gemma4:31b` | Google (via Ollama) | NER (default), GT mapping (HPC) | Default for Stage 1 and GT mapping on HPC |
| `gemma4:31b-cloud` | Google (via Ollama Cloud) | EL (default), GT mapping (local) | Default for Stages 2 and GT mapping on local machine |
| `deepseek-v4-flash` | DeepSeek | RE (default) | Default for Stage 3 |
| `deepseek-v4-pro` | DeepSeek | NER (config template default) | Default model in `.cfg` config files (overridable at CLI) |
| `mistral-large-3:675b` | Mistral | NER (experimental, commented in batch scripts) | — |
| `ministral-3:14b` | Mistral | NER (experimental, commented) | — |
| `nemotron-3-ultra` | NVIDIA | NER (experimental, commented) | — |
| `qwen3.5:397b` | Alibaba/Qwen | NER (experimental, commented) | — |
| `kimi-k2.6` | Moonshot AI | NER (experimental, commented) | — |
| `cogito-2.1:671b` | — | NER (experimental, commented) | — |
| `glm-5.1` | Zhipu AI | NER (experimental, commented) | — |
| `llama4:maverick` | Meta | NER (experimental, commented) | Local only |
| `llama4:scout` | Meta | NER (experimental, commented) | Local only |

All models are accessed via Ollama. Cloud models require `OLLAMA_API_KEY` and use `https://ollama.com`. Local models use `http://localhost:11434`.

---

## Interactive file selection pattern

A consistent interactive file selection pattern is used across many pipeline modules. The pattern works as follows:

1. Recursively scan a pre-configured directory for matching files (NER modules traverse `{model}/{letter}/` subdirectories).
2. If a single file matches, auto-select it silently.
3. If multiple files match, present a numbered menu (showing model/letter context) and prompt for selection.

Modules using this pattern:

| Module | Directory scanned | File pattern |
|---|---|---|
| `ner/ner_evaluate.py` | `ner/ner-output/` (recursive) | `*.spacy` |
| `ner/ner_displacy.py` | `ner/ner-output/` (recursive) + `entity_linking/el-results/` | `*.spacy` (both) |
| `ner/export_ner_to_el_template.py` | `ner/ner-output/` (recursive) | `*.spacy` |
| `entity_linking/el.py` | `ner/ner-output/` (recursive) | `*.spacy` (excluding `_el`) |
| `entity_linking/el_stats.py` | `entity_linking/el-results/` | `*_el.spacy` |
| `entity_linking/el_evaluate.py` | `entity_linking/el-results/` | `*_el.spacy` |
| `relation_extraction/rel_extraction.py` | `entity_linking/el-results/` | `*_el.spacy` |
| `output/tei_exporter.py` | `entity_linking/el-results/` | `*_el.spacy` |
| `output/ato_schacl_rdf_validator.py` | `output/rdf/` | `*_events.ttl` |
| `relation_extraction/query_route.py` | `output/rdf/` | `*_events.ttl` |

---

## Important architectural notes

1. **Hardcoded `INPUT_MODE`**: `ner.py` line 92 sets `INPUT_MODE = "txt"`. To process corrected PageXML from `data/page_updated/`, change this to `"pagexml"`. It is not exposed as a CLI argument.

2. **Custom spaCy extensions**: The extension attributes `kb_id_wikidata_`, `kb_id_geonames_`, and `ent_kb_id_*` are registered independently in 7 different scripts (`el.py`, `rel_extraction.py`, `tei_exporter.py`, `ner_displacy.py`, `el_evaluate.py`, `el_stats.py`, `export_ner_to_el_template.py`). Registration must happen before `DocBin.from_disk()` loading. The `has_extension` guard prevents duplicate registration errors.

3. **PageXML mode vs. TXT mode**: In TXT mode, pages are determined by `[N]` bracket markers or horizontal rules. In PageXML mode, pages come from individual XML files, mapped via `data/1816-scannumber-to-pagenumber.csv`, and a `line_map` (character offset to line index) is produced alongside the offset map.

4. **NER output directory structure**: NER output files are saved to `ner/ner-output/{model}/{letter_label}/` subdirectories (e.g., `ner/ner-output/gemma4:31b/1816/`). Downstream modules recursively scan these subdirectories when discovering NER files.

5. **Fixed output paths**: Several scripts have hardcoded output filenames:
   - `rel_extraction.py` → `output/re/1816_third_letter_events.json` / `output/rdf/1816_third_letter_events.ttl`
   - `tei_exporter.py` → `output/tei/1816_third_letter.tei.xml`
   - `pagexml_enricher.py` → hardcoded spacy/offset/line map input paths

6. **Naming convention**: The label `E22_Human-made_Object` was renamed to `Mode_of_Transportation`. The mention_map in `rel_extraction.py` normalises old names. The tag maps in `ner_evaluate.py` handle this normalisation for evaluation.

7. **EL error propagation**: When an Ollama call fails in EL Stages 2 or 3, the error is propagated as `ERROR:*` strings in the `kb_id` fields, enabling post-hoc analysis rather than silent failure.

8. **NER retry logic**: `ner.py` retries up to 3 times per page on API errors or zero-entity parsing failures. Failed pages are tracked in the `retry_pages_zero_ents` field of the meta JSON.

9. **RE prompt truncation**: `rel_extraction.py` truncates the full prompt at 200,000 characters to stay within the model's context window.

10. **GT mapping scripts**: The `_local.py` and `_hpc.py` versions are near-identical except for model name and timeout. Changes should be applied to both.

11. **Duplicated helper functions**: Roman numeral sorting, marker stripping (`[N]`), offset map discovery, and PageXML namespace extraction are duplicated across multiple modules. Changes to these should be propagated to all copies.

12. **Work-in-progress module**: `output/pagexml_enricher.py` is explicitly marked as WORK IN PROGRESS with hardcoded input paths and limited testing.

13. **No automated tests**: The project has no formal test suite. The only ad-hoc test is `test.sh`, which runs NER on a different input file (`data/1809_sixth_letter.txt`).

---

## File and directory reference

| Path | Type | Module | Description |
|---|---|---|---|
| `ollama_utils.py` | Python | Shared | `stream_ollama_chat()` — shared Ollama streaming utility used by all pipeline stages. Displays thinking tokens and content in real time. |
| `ner/ner.py` | Python | NER | Main NER pipeline script |
| `ner/ner_evaluate.py` | Python | NER | NER evaluation against Recogito ground truth |
| `ner/ner_displacy.py` | Python | NER | Browser-based entity visualization |
| `ner/export_gs_to_el_template.py` | Python | NER | Create EL gold standard CSV from Recogito annotations |
| `ner/export_ner_to_el_template.py` | Python | NER | Create EL annotation CSV from NER output |
| `ner/ner.sh` | Shell | NER | Batch NER runner (single model) |
| `ner/ner_and_eval.sh` | Shell | NER | Batch NER + evaluation runner (multiple models) |
| `ner/config/fewshot.cfg` | Config | NER | spaCy-LLM config: few-shot, Dutch |
| `ner/config/zeroshot.cfg` | Config | NER | spaCy-LLM config: zero-shot, Dutch |
| `ner/config/fewshot_english.cfg` | Config | NER | spaCy-LLM config: few-shot, English |
| `ner/config/zeroshot_english.cfg` | Config | NER | spaCy-LLM config: zero-shot, English |
| `ner/config/prompt_dutch.jinja` | Template | NER | Jinja2 NER prompt (Dutch) |
| `ner/config/prompt_english.jinja` | Template | NER | Jinja2 NER prompt (English) |
| `ner/config/fewshot_examples.yaml` | Config | NER | 13 few-shot NER examples |
| `entity_linking/el.py` | Python | EL | EL pipeline orchestrator |
| `entity_linking/el_candidates.py` | Python | EL | Stage 1: Candidate generation |
| `entity_linking/el_reranker.py` | Python | EL | Stage 2: Candidate reranking |
| `entity_linking/el_selector.py` | Python | EL | Stage 3: Candidate selection |
| `entity_linking/el_evaluate.py` | Python | EL | EL evaluation against gold standard |
| `entity_linking/el_stats.py` | Python | EL | EL statistics display |
| `relation_extraction/rel_extraction.py` | Python | RE | Main RE + RDF generation pipeline |
| `relation_extraction/query_route.py` | Python | RE | Travel route query and display |
| `relation_extraction/validate_rdf.py` | Python | RE | Lightweight ATO schema validation |
| `relation_extraction/re_examples.yaml` | Config | RE | Five few-shot RE examples |
| `relation_extraction/re_prompt_dutch.jinja` | Template | RE | Jinja2 RE prompt (Dutch) |
| `output/tei_exporter.py` | Python | Output | TEI P5 XML document export |
| `output/pagexml_enricher.py` | Python | Output | PageXML annotation enrichment (WORK IN PROGRESS) |
| `output/ato_schacl_rdf_validator.py` | Python | Output | SHACL RDF validation |
| `ground_truth_mapping/gt_to_pagexml_local.py` | Python | GT Map | Local HTR correction |
| `ground_truth_mapping/gt_to_pagexml_hpc.py` | Python | GT Map | HPC HTR correction |
| `SLURM/gt_to_pagexml_mapping.sh` | Shell | GT Map | SLURM batch job for HPC |
| `test.sh` | Shell | — | Quick NER integration test (1809 letter) |

---

## End-to-end running guide

```bash
# 0. Prerequisites
export OLLAMA_API_KEY="your-api-key"
pip install spacy spacy-llm ollama httpx rdflib pyshacl jinja2 pyyaml pandas rapidfuzz requests python-dotenv

# 1. NER (fewshot, Dutch, cloud, default third letter)
python ner/ner.py

# Alternative: zeroshot, English, local Ollama
python ner/ner.py -M zeroshot -H localhost -L english

# Alternative: 1809 letter with horizontal-rule splitting
python ner/ner.py -i data/1809_sixth_letter.txt -S horizontal-rule

# 2. Entity Linking (interactive file picker)
python entity_linking/el.py

# 3. Relation Extraction (interactive file picker)
python relation_extraction/rel_extraction.py

# 4. Output: TEI/XML export (interactive file picker)
python output/tei_exporter.py

# 5. Validation and route query
python relation_extraction/validate_rdf.py
python output/ato_schacl_rdf_validator.py
python relation_extraction/query_route.py
python relation_extraction/query_route.py output/rdf/1816_third_letter_events.ttl --csv

# 6. Evaluation
python ner/ner_evaluate.py
python entity_linking/el_evaluate.py

# 7. Visualization
python ner/ner_displacy.py

# 8. Linking statistics
python entity_linking/el_stats.py

# 9. GT-to-PageXML mapping
python ground_truth_mapping/gt_to_pagexml_local.py

# 10. SLURM job for HPC
sbatch SLURM/gt_to_pagexml_mapping.sh

# 11. Gold standard generation
python ner/export_gs_to_el_template.py
python ner/export_ner_to_el_template.py
```
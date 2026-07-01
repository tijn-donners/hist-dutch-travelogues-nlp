# output/

Stage 4: scholarly-edition export. Produces two editions of the letter: a coordinate-bearing per-page ALTO viewer edition (HTR baselines plus manual transcription plus NER/EL tags, with a shared RDF sidecar) and a loss-free canonical TEI/XML edition (all 313 entities tagged, with embedded RDF). Includes self-checks and a SHACL validator for the ATO RDF.

## Files

- `alto_exporter.py`: export coordinate-bearing ALTO-XML per page. Stage A aligns the manual transcription onto Loghi baselines (LLM line-aligner, cached); Stage B deterministically emits one ALTO 4 doc per page with NER+EL as `<OtherTag>` tags linked to ATO mention URIs. RE/RDF kept in a shared sidecar `.ttl`.
- `tei_exporter.py`: export the canonical TEI/XML edition (single self-contained TEI document, zero entity loss, inline `placeName`/`persName`/`orgName`/`date`/`rs` tags, full ATO RDF embedded as RDF/XML in `<standOff><xenoData>`).
- `pagexml_enricher.py`: legacy/WIP. Writes NER+EL annotations back into PageXML `<Metadata>`.
- `alto_overlay.py`: render a per-page ALTO file as a standalone HTML overlay (scan image background, colour-coded NER labels).
- `alto_selfcheck.py`, `tei_selfcheck.py`: round-trip spot-checks (entity tagging, offset alignment, zero entity loss, well-formedness, determinism).
- `ato_schacl_rdf_validator.py`: validate ATO RDF output against SHACL shapes via pyshacl.

## Subfolders

- `alto/`: per-page ALTO outputs `1816_third_letter_scan00XX.alto.xml` plus `.overlay.html`, page-scan `.jpg` images, `line_alignment.json`, and the shared `1816_third_letter.ttl`.
- `page/`: 10 enriched PageXML files (`0552_0179_0049.xml` ... `0067.xml`, odd scans).
- `rdf/`: 8 per-model `_events.ttl` files plus `ato_schacl_shapes.ttl`.
- `re/`: 8 RE runs, each with `_events.json`, `_events.meta.json`, `_events.think.txt`, and `_mention_map.json`.
- `tei/`: `1816_third_letter.tei.xml`, the canonical edition.

## Running

Export ALTO (defaults point at the gold files):

```
python output/alto_exporter.py
```

Export TEI:

```
python output/tei_exporter.py
```

Render an ALTO overlay (one file, or all 10 if omitted):

```
python output/alto_overlay.py output/alto/1816_third_letter_scan0049.alto.xml
```

Validate the ATO RDF against SHACL shapes:

```
python output/ato_schacl_rdf_validator.py
```

Run the self-checks:

```
python output/alto_selfcheck.py
python output/tei_selfcheck.py
```

There are no CSV files in `output/`; outputs are `.xml`, `.ttl`, `.json`, and `.jpg`.
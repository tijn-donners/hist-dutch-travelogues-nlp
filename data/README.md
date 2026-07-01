# data/

Raw input material for the pipeline: source transcriptions, scanned page images, and Loghi HTR PAGE-XML.

## Files

- `1816_third_letter.txt`: manual transcription of the 1816 third letter, with inline `[N]` page markers. The primary test document.
- `1809_sixth_letter.txt`: manual transcription of the 1809 sixth letter.
- `GT_1816_for_mapping.txt`: full ground-truth 1816 transcription, used by `ground_truth_mapping/` to correct HTR line text.
- `sample_annotations_brief_4_and_5.txt`: Recogito annotation notes for letters 4 and 5.
- `1816-scannumber-to-pagenumber.csv`: maps scan numbers to printed page numbers (see below).
- `page/`: 399 `.png` page scans plus 399 PAGE-XML files from Loghi HTR, named `0552_0179_0001` through `0552_0179_0399` (`.png` + `.xml` pairs). Consumed by `ground_truth_mapping/` and `output/alto_exporter.py`.

## CSV columns

### `1816-scannumber-to-pagenumber.csv`

Maps a sequential scan number to the printed page number (Roman numeral). Used to align PAGE-XML scans with the bracket-delimited transcription pages.

| Column | Description |
|---|---|
| `Scan Number` | Sequential scan index (1, 2, 3, ...) |
| `Page Number` | Printed page number (I, II, III, ...) |
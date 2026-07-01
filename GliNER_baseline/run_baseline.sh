#!/usr/bin/env bash
# Run the GliNER2 baseline, then evaluate it with the existing ner_evaluate.py.
# Usage:
#   bash run_baseline.sh                          # defaults: 1816_third_letter.txt, brackets
#   bash run_baseline.sh --input data/1809_sixth_letter.txt
#   bash run_baseline.sh --threshold 0.4 --split-mode horizontal-rule
# All args are forwarded to gliner_baseline.py.
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Running GliNER2 baseline ==="
# Capture stdout so we can extract the generated .spacy path for evaluation.
OUT=$(python gliner_baseline.py "$@")

echo "$OUT"

# The script prints "Merged DocBin saved to: <path>.spacy" — grab that path.
SPACY_PATH=$(echo "$OUT" | grep -oE '/[^ ]*\.spacy' | head -n1)

if [[ -z "${SPACY_PATH:-}" ]]; then
  echo "ERROR: could not find a .spacy output path in the run output." >&2
  exit 1
fi

echo
echo "=== Evaluating with ner_evaluate.py ==="
python ../ner/ner_evaluate.py -s "$SPACY_PATH"

echo
echo "Scores appended to ../ner/ner-evaluation/scores.csv"
echo "Error log:      ../ner/ner-evaluation/errors_logs/$(basename "$SPACY_PATH" .spacy)_errors.csv"
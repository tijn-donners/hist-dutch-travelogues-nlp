#!/usr/bin/env bash
# Evaluate all .spacy files in ner-output/ against ground truth annotations.
# Runs ner_evaluate.py --spacy-file on each .spacy file found recursively.
# Results (errors CSV + cumulative scores CSV) land in ner-evaluation/.
# Error CSVs go to ner-evaluation/errors_logs/, scores.csv stays in ner-evaluation/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/ner-output"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: $RESULTS_DIR not found"
    exit 1
fi

# Collect all .spacy files
mapfile -t spacy_files < <(find "$RESULTS_DIR" -name "*.spacy" -type f | sort)

if [ ${#spacy_files[@]} -eq 0 ]; then
    echo "No .spacy files found in $RESULTS_DIR"
    exit 1
fi

echo "Found ${#spacy_files[@]} .spacy files to evaluate"
echo "=========================================="
echo ""

count=0
failed=0
for spacy_file in "${spacy_files[@]}"; do
    count=$((count + 1))
    rel="${spacy_file#$RESULTS_DIR/}"
    echo "[$count/${#spacy_files[@]}] Evaluating: $rel"
    echo "----------------------------------------"
    if python "$SCRIPT_DIR/ner_evaluate.py" --spacy-file "$spacy_file"; then
        echo ""
        echo "  ✓ Done"
    else
        echo ""
        echo "  ✗ Failed (exit code $?)"
        failed=$((failed + 1))
    fi
    echo "=========================================="
    echo ""
done

echo "All done — $count evaluated, $failed failed"
echo "Scores CSV: $SCRIPT_DIR/ner-evaluation/scores.csv"

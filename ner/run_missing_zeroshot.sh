### This script runs the missing zeroshot NER configs that have not been run yet.
### It checks which .spacy files are missing and runs only those combinations.
### Run from any directory — paths resolve relative to this script's location.
###
### Logs go to ner-evaluation/errors_logs/
### Evaluation results are appended to ner-evaluation/scores.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
mkdir -p ner-evaluation/errors_logs

### Configuration

INPUT_FILE="data/1816_third_letter.txt"
LETTER_LABEL="$(basename "$INPUT_FILE" .txt)"
SHORT_LABEL="${LETTER_LABEL%%_*}"
SPLIT_MODE="brackets"
OLLAMA_HOST="cloud"
MODE="zeroshot"

# All models, languages and temperatures (zeroshot only)
MODELS=(
    "deepseek-v4-pro"
    "qwen3.5:397b"
    "kimi-k2.7-code"
    "glm-5.1"
)

LANGUAGES=("dutch" "english")
TEMPERATURES=(0.0 0.5 1.0)

TOTAL_TODO=0
TOTAL_SKIP=0

for MODEL in "${MODELS[@]}"; do
    for LANG in "${LANGUAGES[@]}"; do
        for TEMP in "${TEMPERATURES[@]}"; do
            SPACY_FILE="ner-output/${MODEL}/${SHORT_LABEL}/${LETTER_LABEL}__${MODEL}_t${TEMP}_${MODE}_${LANG}.spacy"

            if [ -f "$SPACY_FILE" ]; then
                echo "=== SKIP (already exists): $SPACY_FILE ==="
                TOTAL_SKIP=$((TOTAL_SKIP + 1))
                continue
            fi

            TOTAL_TODO=$((TOTAL_TODO + 1))

            echo ""
            echo "============================================================"
            echo "=== ${TOTAL_TODO}. Running: ${MODEL} | ${LANG} | t=${TEMP} ==="
            echo "============================================================"
            echo "model: $MODEL"
            echo "source file: $INPUT_FILE"
            echo "temperature: $TEMP"
            echo "prompt method: $MODE"
            echo "prompt language: $LANG"
            echo "ollama host: $OLLAMA_HOST"
            echo "pages split on: $SPLIT_MODE"

            LOG_FILE="ner-evaluation/errors_logs/${LETTER_LABEL}__${MODEL}_t${TEMP}_${MODE}_ner.log"
            python3 ner.py -i "$INPUT_FILE" -m "$MODEL" -M "$MODE" -t "$TEMP" \
                -H "$OLLAMA_HOST" -S "$SPLIT_MODE" --language "$LANG" 2>&1 | tee "$LOG_FILE"

            if [ -f "$SPACY_FILE" ]; then
                echo "=== Evaluating: $SPACY_FILE ==="
                python3 ner_evaluate.py -s "$SPACY_FILE"
            else
                echo "=== SKIP (spacy file not found after run): $SPACY_FILE ==="
            fi
        done
    done
done

echo ""
echo "============================================================"
echo "=== Done: ${TOTAL_TODO} runs completed, ${TOTAL_SKIP} skipped (already existed) ==="
echo "============================================================"
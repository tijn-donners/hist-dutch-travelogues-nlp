### This script runs the NER pipeline across all fewshot configurations:
###   - every model × every prompt language (dutch/english) × every temperature (0.0, 0.5, 1.0)
### Overwrites existing fewshot .spacy files (those were run with old fewshot examples).
### Uses Ollama's cloud API via an API key.
### Run from any directory — paths resolve relative to this script's location.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
mkdir -p ner-evaluation/errors_logs

### --- Configuration ---

INPUT_FILE="data/1816_third_letter.txt"
LETTER_LABEL="$(basename "$INPUT_FILE" .txt)"
SHORT_LABEL="${LETTER_LABEL%%_*}"  # first segment before underscore, e.g. "1816"
SPLIT_MODE="brackets"
OLLAMA_HOST="cloud"
MODE="fewshot"

# Language × temperature grid
LANGS=("dutch" "english")
TEMPS=("0.0" "0.5" "1.0")

# Models to run — add or remove as needed
MODELS=(
    "gemma4:31b"
    "deepseek-v4-pro"
    "deepseek-v4-flash"
    "mistral-large-3:675b"
    "qwen3.5:397b"
    "kimi-k2.7-code"
    "cogito-2.1:671b"
    "glm-5.1"
)

### --- Run grid ---

for MODEL in "${MODELS[@]}"; do
    for LANG in "${LANGS[@]}"; do
        for TEMP in "${TEMPS[@]}"; do

            echo "=============================================="
            echo "  MODEL:      $MODEL"
            echo "  LANGUAGE:   $LANG"
            echo "  TEMP:       $TEMP"
            echo "  MODE:       $MODE"
            echo "  HOST:       $OLLAMA_HOST"
            echo "  SPLIT:      $SPLIT_MODE"
            echo "=============================================="

            LOG_FILE="ner-evaluation/errors_logs/${LETTER_LABEL}__${MODEL}_t${TEMP}_${MODE}_${LANG}.log"
            python3 ner.py \
                -i "$INPUT_FILE" \
                -m "$MODEL" \
                -M "$MODE" \
                -t "$TEMP" \
                -H "$OLLAMA_HOST" \
                -S "$SPLIT_MODE" \
                --language "$LANG" \
                2>&1 | tee "$LOG_FILE"

            SPACY_FILE="ner-output/${MODEL}/${SHORT_LABEL}/${LETTER_LABEL}__${MODEL}_t${TEMP}_${MODE}_${LANG}.spacy"
            if [ -f "$SPACY_FILE" ]; then
                echo "=== Evaluating: $SPACY_FILE ==="
                python3 ner_evaluate.py -s "$SPACY_FILE"
            else
                echo "=== SKIP (spacy file not found): $SPACY_FILE ==="
            fi

        done
    done
done

echo "=== All fewshot grid runs complete ==="

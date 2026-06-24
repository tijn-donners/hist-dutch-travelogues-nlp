### This script runs the NER pipeline on the defined models 
### This current configuration uses Ollama's cloud models via an API key
### Run from any directory — paths resolve relative to this script's location

### Logs go to ner-evaluation/errors_logs/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
mkdir -p ner-evaluation/errors_logs

### Configuration

INPUT_FILE="data/1816_third_letter.txt"
LETTER_LABEL="$(basename "$INPUT_FILE" .txt)"
SHORT_LABEL="${LETTER_LABEL%%_*}"  # first segment before underscore, e.g. "1816"
SPLIT_MODE="brackets" # "brackets" for [N] markers or "horizontal-rule" for underscore-separated pages
OLLAMA_HOST="cloud"
# "cloud" for Ollama hosted API or "localhost" for using Ollama models locally.
# Only use "localhost" on HPC or machines that can run large models locally.
# Make sure to download the LLMs you want to use for the NER pipeline.
# This script will pull the LLMs if not yet downloaded (might increase job time for SLURM jobs).
LANG="english"

# LLM configuration
MODE="fewshot" # 'zeroshot' or 'fewshot'
TEMP="0.0" # float from 0.0 to 1.0

# This script iterates over all selected models and performs NER and calculates precision, recall and f1 for each model
# Uncomment, add or remove Ollama models you want run the pipeline with below
MODELS=(
# "gemma4:31b"
    # "deepseek-v4-pro"
    # "deepseek-v4-flash"
    # "mistral-large-3:675b" 
    # "qwen3.5:397b"
    # "kimi-k2.7-code" 
    "cogito-2.1:671b"
    # "glm-5.1"
    )

for MODEL in "${MODELS[@]}"; do
    if [ "$OLLAMA_HOST" = "localhost" ]; then
        # Ensure local Ollama server is running
        if ! pgrep -x ollama > /dev/null; then
            echo "=== Starting Ollama server ==="
            ollama serve &
            sleep 4
        fi
        echo "=== Pulling model: $MODEL ==="
        ollama pull "$MODEL"
    fi

    echo "=== Running NER ==="
    echo "model: $MODEL"
    echo "source file: $INPUT_FILE"
    echo "temperature: $TEMP"
    echo "prompt method: $MODE"
    echo "prompt language: $LANG"
    echo "ollama host: $OLLAMA_HOST"
    echo "pages are split on: $SPLIT_MODE"

    LOG_FILE="ner-evaluation/errors_logs/${LETTER_LABEL}__${MODEL}_t${TEMP}_${MODE}_ner.log"
    python3 ner.py -i "$INPUT_FILE" -m "$MODEL" -M "$MODE" -t "$TEMP" -H "$OLLAMA_HOST" -S "$SPLIT_MODE" --language "$LANG" 2>&1 | tee "$LOG_FILE"

    SPACY_FILE="ner-output/${MODEL}/${SHORT_LABEL}/${LETTER_LABEL}__${MODEL}_t${TEMP}_${MODE}_${LANG}.spacy"
    if [ -f "$SPACY_FILE" ]; then
        echo "=== Evaluating: $SPACY_FILE ==="
        python3 ner_evaluate.py -s "$SPACY_FILE"
    else
        echo "=== SKIP (spacy file not found): $SPACY_FILE ==="
    fi
done

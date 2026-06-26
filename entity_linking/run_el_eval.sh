#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Run the EL pipeline on a .spacy file and evaluate the results.
#
# Usage:
#   bash entity_linking/run_el.sh --input ner/ner-output/PATH/file.spacy \\
#                                 --model gemma4:31b \\
#                                 --temperature 0.1
#
# All arguments are optional; defaults are shown below.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
INPUT_FILE=""
MODEL="deepseek-v4-flash"
TEMPERATURE=0.0
TOP_K=3
# Think mode (controls reasoning/thinking token budget for thinking models):
#   ""       (empty) → model's own default (kimi: thinking on, full intensity)
#   "false"  → thinking disabled entirely, no [thinking] tokens, faster
#   "low"    → thinking enabled but constrained to minimal reasoning
#   "medium" → moderate thinking intensity
#   "high"   → maximum thinking intensity
THINK="false"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)    INPUT_FILE="$2";  shift 2 ;;
    --model)    MODEL="$2";       shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-k)    TOP_K="$2";       shift 2 ;;
    --think)    THINK="$2";       shift 2 ;;
    --help|-h)  sed -n '3,10p' "$0"; exit 0 ;;
    *)          echo "Unknown: $1"; exit 1 ;;
  esac
done

# ── Pick input file if not given ──────────────────────────────────────────────
if [ -z "$INPUT_FILE" ]; then
  echo "Available .spacy files in ner/ner-output/:"
  mapfile -t FILES < <(find "$PROJECT_DIR/ner/ner-output" -name '*.spacy' ! -name '*_el.spacy' | sort)
  if [ ${#FILES[@]} -eq 0 ]; then
    echo "No .spacy files found."
    exit 1
  fi
  for i in "${!FILES[@]}"; do
    rel="${FILES[$i]#$PROJECT_DIR/}"
    echo "  [$((i+1))] $rel"
  done
  read -r -p "Select number: " SEL
  INPUT_FILE="${FILES[$((SEL-1))]}"
  if [ -z "$INPUT_FILE" ]; then
    echo "Invalid selection."
    exit 1
  fi
fi

INPUT_FILE="$(realpath "$INPUT_FILE")"
if [ ! -f "$INPUT_FILE" ]; then
  echo "File not found: $INPUT_FILE"
  exit 1
fi

echo ""
echo "═══ EL Pipeline ═══"
echo "  Input:       $INPUT_FILE"
echo "  Model:       $MODEL"
echo "  Temperature: $TEMPERATURE"
echo "  Top-k:       $TOP_K"
if [ -n "$THINK" ]; then
  echo "  Think:       $THINK"
fi
echo ""

# ── Run EL pipeline ──────────────────────────────────────────────────────────
# Call link_entities() directly with the given parameters, then save results.
python3 << PYEOF
import sys, json, shutil, time
from pathlib import Path

sys.path.insert(0, "$SCRIPT_DIR")
from el import link_entities, clear_cache
from spacy.tokens import DocBin

INPUT = Path("$INPUT_FILE")
MODEL = "$MODEL"
TEMPERATURE = float("$TEMPERATURE")
TOP_K = int("$TOP_K")
THINK = "$THINK" or None  # empty string → None (model default)
if THINK is not None and THINK.lower() == "false":
    THINK = False

# Determine cloud vs local
from el import OLLAMA_URL, OLLAMA_HEADERS

clear_cache()

t0 = time.time()
docs, stats = link_entities(
    spacy_file=str(INPUT),
    model_name=MODEL,
    top_k_rerank=TOP_K,
    ollama_url=OLLAMA_URL,
    ollama_headers=OLLAMA_HEADERS,
    think=THINK,
    temperature=TEMPERATURE,
)
t1 = time.time()
duration = t1 - t0

# Save output
OUT_DIR = Path("$SCRIPT_DIR/el-results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

stem = INPUT.stem
model_slug = MODEL.replace(":", "-").replace("/", "-")
think_slug = f"_think{str(THINK).lower()}" if THINK is not None else ""
output_path = OUT_DIR / f"{stem}__{model_slug}_t{TEMPERATURE}{think_slug}_el.spacy"
docbin = DocBin(docs=docs, store_user_data=True)
docbin.to_disk(str(output_path))
print(f"\nSaved to: {output_path}")

# Copy offset map with the same stem
offset_stem = stem + "_offset_map"
offset_map = INPUT.parent / f"{offset_stem}.json"
if offset_map.exists():
    dest = OUT_DIR / f"{stem}__{model_slug}_t{TEMPERATURE}{think_slug}_offset_map.json"
    shutil.copy2(offset_map, dest)
    print(f"Offset map copied to: {dest}")
else:
    print(f"Warning: no offset map found ({offset_map.name})")

# Save stats + duration to a temp JSON for the evaluation step
info = {"stats": stats, "duration": duration, "model": MODEL, "temperature": TEMPERATURE, "top_k": TOP_K, "think": str(THINK).lower() if THINK is not None else ""}
info_path = OUT_DIR / f"{stem}__{model_slug}_t{TEMPERATURE}{think_slug}_el_run_info.json"
with open(info_path, "w") as f:
    json.dump(info, f)
print(f"Run info saved to: {info_path}")
PYEOF

# ── Run evaluation ─────────────────────────────────────────────────────────────
# Determine the output filename that was just created
MODEL_SLUG="${MODEL//:/-}"
MODEL_SLUG="${MODEL_SLUG//\//-}"
THINK_SLUG=""
if [ -n "$THINK" ]; then
  THINK_SLUG="_think${THINK}"
fi
OUTPUT_FILE="$SCRIPT_DIR/el-results/$(basename "${INPUT_FILE%.spacy}")__${MODEL_SLUG}_t${TEMPERATURE}${THINK_SLUG}_el.spacy"
RUN_INFO="$SCRIPT_DIR/el-results/$(basename "${INPUT_FILE%.spacy}")__${MODEL_SLUG}_t${TEMPERATURE}${THINK_SLUG}_el_run_info.json"

if [ ! -f "$OUTPUT_FILE" ]; then
  echo "Output not found: $OUTPUT_FILE"
  exit 1
fi

# Read run info
STATS_JSON=$(python3 -c "import json; print(json.dumps(json.load(open('$RUN_INFO'))['stats']))")
DURATION=$(python3 -c "import json; print(json.load(open('$RUN_INFO'))['duration'])")

# Determine inference type
if [ -n "${OLLAMA_API_KEY:-}" ]; then
  INFERENCE_TYPE="cloud"
else
  INFERENCE_TYPE="local"
fi

echo ""
echo "═══ Evaluation ═══"
echo "  EL file:     $OUTPUT_FILE"
echo "  Inference:   $INFERENCE_TYPE"
echo ""

python3 "$SCRIPT_DIR/el_evaluate.py" \
  --el "$OUTPUT_FILE" \
  --pipeline-stats "$STATS_JSON" \
  --duration "$DURATION" \
  --model "$MODEL" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --think "$THINK" \
  --inference-type "$INFERENCE_TYPE"

# run_info.json kept for batch_evaluate.py to reuse

echo ""
echo "Done."

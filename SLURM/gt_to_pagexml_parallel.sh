#!/bin/bash
#SBATCH --job-name=gt_pagexml_mapping
#SBATCH --output=gt_%j.log
#SBATCH --error=gt_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8          # enough cores for 4 parallel workers + Ollama threads
#SBATCH --mem=48G                  # extra headroom for parallel XML parsing
#SBATCH --time=08:00:00            # realistic ceiling; job will likely finish in ~2h
#SBATCH --gpus-per-node=1

# ── Ollama config ────────────────────────────────────────────────────────────
export OLLAMA_MODELS=/scratch/$USER/ollama_models_scratch

# Allow Ollama to serve this many requests concurrently.
# Must match MAX_WORKERS in the Python script.
export OLLAMA_NUM_PARALLEL=4

# Keep the model in VRAM between requests (seconds; 0 = never unload)
export OLLAMA_KEEP_ALIVE=3600

# Let Ollama use all available CPU threads for non-GPU work
export OLLAMA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── Start Ollama server ───────────────────────────────────────────────────────
/scratch/$USER/ollama/bin/ollama serve &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"

# Wait until the server is actually ready (up to 30 s)
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/ > /dev/null 2>&1; then
        echo "Ollama ready after ${i}s"
        break
    fi
    sleep 1
done

# ── Model warm-up ─────────────────────────────────────────────────────────────
# NOTE: Run `ollama pull gemma4:31b` manually once before submitting this job.
# That saves ~5 minutes of pull time on every run.
echo "Warming model into GPU VRAM…"
/scratch/$USER/ollama/bin/ollama run gemma4:31b "hello" --nowordwrap
echo "Model loaded."

# ── Python environment ────────────────────────────────────────────────────────
source $HOME/venvs/ollama/bin/activate
# Remove the pip install line — run it once locally instead:
#   pip install pandas ollama

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Starting Python script…"
python3 -u /scratch/$USER/hist-dutch-travelogues-nlp/gt_to_pagexml_parallel.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
kill $OLLAMA_PID 2>/dev/null
echo "Done."

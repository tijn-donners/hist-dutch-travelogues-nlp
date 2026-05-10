#!/bin/bash
#SBATCH --job-name=gt_pagexml_mapping
#SBATCH --output=gt_%j.log
#SBATCH --error=gt_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=1

export OLLAMA_MODELS=/scratch/$USER/ollama_models_scratch

# Start Ollama (use the binary rather than the module, since it is very outdated)server in background
/scratch/$USER/ollama/bin/ollama serve &
sleep 15  # wait for server to start
/scratch/$USER/ollama/bin/ollama pull gemma4:31b
/scratch/$USER/ollama/bin/ollama ls

echo "Loading model into GPU memory..."
/scratch/$USER/ollama/bin/ollama run gemma4:31b "hello" --nowordwrap
echo "Model loaded, starting Python script..."

source $HOME/venvs/ollama/bin/activate
pip install pandas ollama

python3 -u /scratch/s6437265/hist-dutch-travelogues-nlp/gt_to_pagexml.py
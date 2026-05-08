#!/bin/bash
#SBATCH --job-name=gt_pagexml_mapping
#SBATCH --output=gt_%j.log
#SBATCH --error=gt_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1

module Python/3.11.3-GCCcore-12.3.0

export OLLAMA_MODELS=/scratch/$USER/ollama_models_scratch

# Start Ollama (use the binary rather than the module, since it is very outdated)server in background
/scratch/$USER/ollama/bin serve &
sleep 15  # wait for server to start
/scratch/$USER/ollama/bin pull gemma4:31b
/scratch/$USER/ollama/bin ls

echo "Loading model into GPU memory..."
/scratch/$USER/ollama/bin run gemma4:31b "hello" --nowordwrap
echo "Model loaded, starting Python script..."

source $HOME/venvs/ollama/bin/activate
pip install pandas ollama

python3 /scratch/s6437265/hist-dutch-travelogues-nlp/gt_to_pagexml.py
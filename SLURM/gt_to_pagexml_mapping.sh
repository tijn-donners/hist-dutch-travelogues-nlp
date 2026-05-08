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

module load ollama/0.6.0-GCCcore-13.3.0-CUDA-12.6.0
module load Python/3.11.3-GCCcore-13.3.0

export OLLAMA_MODELS=/scratch/$USER/ollama_models_scratch

# Start Ollama server in background
ollama serve
sleep 15  # wait for server to start

#ollama pull gemma4:31b

source $HOME/venvs/ollama/bin/activate
pip install pandas ollama

python3 /scratch/s6437265/hist-dutch-travelogues-nlp/gt_to_pagexml.py
# SLURM/

HPC batch submission for the ground-truth-mapping HTR-correction job.

## Files

- `gt_to_pagexml_mapping.sh`: SLURM batch script that submits `ground_truth_mapping/gt_to_pagexml_hpc.py` to the HPC cluster (gpu partition, 32G memory, 1 GPU). It sets `OLLAMA_MODELS=/scratch/$USER/ollama_models_scratch` for the local model cache.

## Running

```
sbatch SLURM/gt_to_pagexml_mapping.sh
```

There are no CSV files in this folder.
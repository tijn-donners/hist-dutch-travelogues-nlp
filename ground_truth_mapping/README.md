# ground_truth_mapping/

HTR correction. Corrects noisy Loghi HTR line text in the `data/page/*.xml` PAGE-XML files by aligning each HTR line to its correct substring of the ground-truth transcription (`data/GT_1816_for_mapping.txt`), using an Ollama LLM to do the line-to-GT mapping. Pages without a scan-to-page mapping (from `data/1816-scannumber-to-pagenumber.csv`) or without GT content are copied as-is. Output goes to `data/page_updated/`, with summary and unmatched-line CSV reports.

## Files

- `gt_to_pagexml_local.py`: local version. Uses the Ollama cloud model `gemma4:31b-cloud`.
- `gt_to_pagexml_hpc.py`: HPC version. Uses a local Ollama model `gemma4:31b`.

## Running

Local (Ollama cloud):

```
python ground_truth_mapping/gt_to_pagexml_local.py
```

On the HPC cluster, submit via SLURM (see [`SLURM/README.md`](../SLURM/README.md)):

```
sbatch SLURM/gt_to_pagexml_mapping.sh
```

There are no CSV files in this folder.
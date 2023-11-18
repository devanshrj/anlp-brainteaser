# BrainTeaser

Notes:
- The training and evaluation data for the ***Sentence Puzzle*** task is in `data`, converted from `.npy` (original data format) to `.json` using `load_data.py`.
- `roberta-baseline.py` and `roberta-cskg-baseline.py` were used to reproduce RoBERTa-L and RoBERTa-L (CSKG) results respectively.
- `flant5-baseline.py` was used to reproduce FlanT5 (780M, 3B, and 11B) results.
- `Metrics_Calculation.ipynb` was used to calculate metrics.
- RoBERTa baselines are based on the code for the paper [Knowledge-driven Data Construction for Zero-shot Evaluation in Commonsense Question Answering](https://github.com/Mayer123/HyKAS-CSKG/tree/main). Weights for RoBERTa (CSKG) were also obtained from the same repository.

Steps to run:
- Requirements: `transformers[torch]`
- RoBERTa baseline: `python roberta-baseline.py --dataset_file data/SP-train.json --out_dir roberta-l-results --device 0`
- RoBERTa (CSKG) baseline: `python roberta-cskg-baseline.py --dataset_file data/SP-train.json --out_dir roberta-cskg-results --device 0 --lm roberta_cskg`
- FlanT5 baseline: `python flant5-baseline.py`
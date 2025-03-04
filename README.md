# Downstream Trade-offs of a Family of Text-Watermarks

This repository contains the code for our EMNLP Findings 2024 paper [Downstream Trade-offs of a Family of Text-Watermarks](https://aclanthology.org/2024.findings-emnlp.821/). We empirically quantify the adverse downstream impacts of the KGW family of text watermarks for autoregressive language models over a set of tasks involving classification, multiple-choice question-answering, and generation. Our codebase currently supports evaluations of the [KGW](https://openreview.net/forum?id=aX8ig9X2a7), [SIR](https://openreview.net/forum?id=6p8lpe4MNf) and [EWD](https://aclanthology.org/2024.acl-long.630/) watermark algorithms when applied to OPT 6.7B, LLaMA 7B and Mistral 7B. 

## Requirements
Please use the latest versions of `nltk`, `tqdm`, `torch`, `jieba`, `accelerate`, and `transformers`. This repository was tested using `nltk==3.9.1`, `tqdm==4.67.1`, `torch==2.5.1`, `jieba==0.42.1`, `accelerate==1.4.0` and `transformers==4.49.0`.

You can complete this setup using:

```pip install -r requirements.txt```

## Code Structure 
`watermark/` contains implementations of [KGW](https://openreview.net/forum?id=aX8ig9X2a7), [SIR](https://openreview.net/forum?id=6p8lpe4MNf) and [EWD](https://aclanthology.org/2024.acl-long.630/) watermarks equipped with helper functions that allow for easy calibration and downstream evaluation. Some associated configuration defaults for these algorithms are provided at `config/`.

`calibration/` contains the scripts relevant for isolating pareto-optimal watermark hyperparameters (see section 3.2 of our paper). These scripts assume that calibration dataset is present at `data/calibration/processed_c4.json`.

`evaluation/` contains code for performing downstream evaluations. The codebase is currently configured to read these downstream datasets from `data/downstream/`. We currently support downstream evaluations over the following datasets: BoolQ, SST-2, CB, HellaSwag, PIQA, SQuAD2, DROP, WMT14-En-Fr and WMT20-En-De. On completing execution, the outputs and scores are stored inside `outputs/` by default.

Under `calibration_results`, we provide the outputs of the calibration procedure for models OPT 6.7B, LLaMA 7B and Mistral 7B for each of the supported watermark algorithms so that you may use them for quick downstream experimentation. 

`utils/` contains utilities for file I/O and model loading. `exceptions/` contains utilities for error-handling. 

## Usage
You can use `calibration/isolate_delta.py` to find pareto-optimal hyperparameters for a specified model and algorithm using the following syntax:
```
python3 -m calibration.isolate_delta --model {opt,llama,mistral} --algorithm {KGW,SIR,EWD} --gamma GAMMA [--target_tpr TARGET_TPR] [--target_fpr TARGET_FPR] [--num_examples NUM_EXAMPLES] [--generation_length GENERATION_LENGTH] [--output_dir OUTPUT_DIR]
```

Once you have run this script for all the gamma values you wish to consider for pareto-optimal calibration, you can use `evaluation/downstream_eval.py` to perform downstream evaluations. The script automatically chooses the pareto-optimal `(gamma, delta)` setting to use by scanning the calibration results. 

```
python3 -m evaluation.downstream_eval --model {opt,llama,mistral} --algorithm {KGW,SIR,EWD} --dataset {boolq,sst2,cb,hellaswag,piqa,drop,squad2,wmt14enfr,wmt20ende} [--target_tpr TARGET_TPR] [--target_fpr TARGET_FPR] [--calibration_results_dir CALIBRATION_RESULTS_DIR] [--num_examples NUM_EXAMPLES] [--seed SEED] [--scores_dir SCORES_DIR] [--output_dir OUTPUT_DIR]

```

You can also directly specify `gamma` and `delta` as arguments for ad-hoc evaluations.

```
python3 -m evaluation.downstream_eval --model {opt,llama,mistral} --algorithm {KGW,SIR,EWD} --dataset {boolq,sst2,cb,hellaswag,piqa,drop,squad2,wmt14enfr,wmt20ende} --gamma GAMMA --delta DELTA [--num_examples NUM_EXAMPLES] [--seed SEED] [--scores_dir SCORES_DIR] [--output_dir OUTPUT_DIR]
```

**Note:** Since Mistral 7B is a gated model, you may need to apply for access at https://huggingface.co/mistralai/Mistral-7B-v0.1 before you can use it. Please set up an [access token](https://huggingface.co/docs/hub/en/security-tokens) and use [HuggingFace CLI](https://huggingface.co/docs/hub/en/security-tokens) to log in before attempting to access the model using the above scripts.

## Citation
If you wish to cite our work, please use
```
@inproceedings{ajith-etal-2024-downstream,
    title = "Downstream Trade-offs of a Family of Text Watermarks",
    author = "Ajith, Anirudh and Singh, Sameer and Pruthi, Danish",
    editor = "Al-Onaizan, Yaser and Bansal, Mohit and Chen Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.821/",
    doi = "10.18653/v1/2024.findings-emnlp.821",
    pages = "14039--14053"
}
```

## Acknowledgements
Parts of this repository were adapted from https://github.com/THU-BPM/MarkLLM. The data we used for downstream evalutation and calibration purposes was generated using https://github.com/EleutherAI/lm-evaluation-harness. We have made this data directly available in this repository under `data/`.

## Bugs or Questions?
If you have any questions relating to the paper or repository, please don't hesitate to reach out to anirudh.ajith@gmail.com.

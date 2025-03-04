# ===============================================================
# downstream_eval.py
# Description: Evaluate downstream performance for a given set of 
#              hyperparameters, model and algorithm
# ===============================================================

import os
import json
import argparse
import numpy as np

from utils.utils import add_evaluation_record_to_file
from utils.model_loading import ModelAndTokenizer
from evaluation.dataset import load_dataset, evaluate_dataset


def read_pareto_optimal_gamma_delta(model, algorithm, tpr, fpr, calibration_results_dir):
    calibration_results_path = os.path.join(calibration_results_dir, f'{model}_{algorithm}.json')
    with open(calibration_results_path, 'r') as f:
        calibration_results = json.load(f)
    
    gamma_delta_ppl_tuples = []
    for result in calibration_results:
        if result["target_tpr"] == tpr and result["target_fpr"] == fpr:
            gamma_delta_ppl_tuples.append((result["gamma"], result["delta"], result["perplexity"]))
    if len(gamma_delta_ppl_tuples) == 0:
        raise ValueError(f"No entries found for tpr: {tpr}, fpr: {fpr} in {calibration_results_path}")
        
    best_gamma, best_delta, _ = min(gamma_delta_ppl_tuples, key=lambda x: x[2])
    return best_gamma, best_delta

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True) # 'opt', 'llama', 'mistral
    parser.add_argument('--algorithm', type=str, required=True) # 'KGW', 'SIR', 'EWD'
    parser.add_argument('--dataset', type=str, required=True) # boolq, sst2, cb, hellaswag, piqa, drop, squad2, wmt14enfr, wmt20ende
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--delta', type=float, default=None)
    parser.add_argument('--target_tpr', type=float, default=0.75)
    parser.add_argument('--target_fpr', type=float, default=0.01)
    parser.add_argument('--calibration_results_dir', type=str, default='calibration_results')
    parser.add_argument('--num_examples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default="outputs")
    args = parser.parse_args()

    print("Evaluating downstream performance for args:", args)
    model_and_tokenizer = ModelAndTokenizer(args.model)
    model = model_and_tokenizer.get_model()
    tokenizer = model_and_tokenizer.get_tokenizer()
    device = "cuda" if model_and_tokenizer.use_gpu else "cpu"

    dataset = load_dataset(args.dataset, args.seed, args.num_examples)

    if args.gamma is None:
        print("gamma argument not provided.")
        print(f"Reading pareto optimal gamma, delta from {args.calibration_results_dir} for tpr: {args.target_tpr}, fpr: {args.target_fpr}")
        gamma, delta = read_pareto_optimal_gamma_delta(args.model, args.algorithm, args.target_tpr, args.target_fpr, args.calibration_results_dir)
    else:
        gamma, delta = args.gamma, args.delta
    print("Using gamma:", gamma, "delta:", delta)

    mean_score, outputs = evaluate_dataset(
        model, tokenizer, device, 
        args.algorithm, gamma, delta, 
        dataset, args.dataset
    )

    print(f"Mean score: {mean_score}\n")
    
    output_path = os.path.join(args.output_dir, f'{args.model}_{args.algorithm}_{args.dataset}.json')

    print(f"Savings outputs to {output_path}")
    add_evaluation_record_to_file(output_path, {
        "gamma": gamma,
        "delta": delta,
        "outputs": outputs
    })
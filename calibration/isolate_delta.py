# ==============================================================
# isolate_delta.py
# Description: Find appropriate delta, and associated perplexity 
#              for a given gamma, target TPR and FPR
# ==============================================================

import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from utils.transformers_config import TransformersConfig
from utils.model_loading import ModelAndTokenizer
from utils.utils import add_evaluation_record_to_file
from watermark.auto_watermark import AutoWatermark
from calibration.dataset import C4Dataset
from calibration.tools.text_editor import TruncatePromptTextEditor
from calibration.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from calibration.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline

def assess_tpr(model, tokenizer, device, algorithm, gamma, delta, target_fpr, num_examples, generation_length):
    calibration_dataset = C4Dataset('data/calibration/processed_c4.json', num_examples=num_examples)
    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=model.config.vocab_size,
        device=device,
        max_new_tokens=generation_length,
        min_length=generation_length + 30,
        do_sample=False,
        no_repeat_ngram_size=4
    )
    
    watermark = AutoWatermark.load(
        algorithm, gamma, delta, 
        algorithm_config=os.path.join('config', f'{algorithm}.json'),
        transformers_config=transformers_config
    )

    w_pipline = WatermarkedTextDetectionPipeline(calibration_dataset, text_editor_list=[TruncatePromptTextEditor()])
    uw_pipline = UnWatermarkedTextDetectionPipeline(calibration_dataset, text_editor_list=[])
    w_evaluations = w_pipline.evaluate(watermark)
    uw_evaluations = uw_pipline.evaluate(watermark)

    calculator = DynamicThresholdSuccessRateCalculator(["TPR"], "target_fpr", target_fpr)
    tpr = calculator.calculate(w_evaluations, uw_evaluations)["TPR"]
    return tpr

def find_delta(model, tokenizer, device, algorithm, gamma, target_tpr, target_fpr, num_examples, generation_length):
    delta_min, delta_max = 0.0, 10.0
    print("Initializing search range: δ ∈ [0.0, 10.0]")
    tpr_max = assess_tpr(model, tokenizer, device, algorithm, gamma, delta_max, target_fpr, num_examples, generation_length)
    print(f'delta: {delta_max}, tpr: {tpr_max}')
    while tpr_max < target_tpr:
        delta_min = delta_max
        delta_max = delta_max * 2
        print(f"Updated search range: δ ∈ [{delta_min}, {delta_max}]")
    
    while delta_max - delta_min > 0.01:
        delta = (delta_min + delta_max) / 2
        tpr = assess_tpr(model, tokenizer, device, algorithm, gamma, delta, target_fpr, num_examples, generation_length)
        print(f'delta: {delta}, tpr: {tpr}')
        if tpr > target_tpr:
            delta_max = delta
        else:
            delta_min = delta
        print(f"Updated search range: δ ∈ [{delta_min}, {delta_max}]")
        if abs(tpr - target_tpr) < 0.01:
            break
    return delta

def assess_perplexity(model, tokenizer, device, algorithm, gamma, delta, num_examples):
    perplexity_dataset = C4Dataset('data/calibration/processed_c4.json', num_examples=num_examples)

    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=model.config.vocab_size,
        device=device,
        max_new_tokens=100,
        min_length=100 + 30,
        do_sample=False,
        no_repeat_ngram_size=4
    )

    watermark = AutoWatermark.load(
        algorithm, gamma, delta, 
        algorithm_config=os.path.join('config', f'{algorithm}.json'),
        transformers_config=transformers_config
    )

    perplexities = []
    for example_idx in tqdm(range(len(perplexity_dataset)), desc="Evaluating perplexity"):
        prompt = perplexity_dataset.get_prompt(example_idx)
        perplexity = watermark.get_perplexity(prompt)
        perplexities.append(perplexity)
    mean_perplexity = np.mean(perplexities)
    return mean_perplexity

def check_if_already_computed(model, algorithm, gamma, tpr, fpr, output_dir):
    calibration_results_path = os.path.join(output_dir, f'{model}_{algorithm}.json')
    if os.path.exists(calibration_results_path):
        with open(calibration_results_path, 'r') as f:
            calibration_results = json.load(f)
        
        for result in calibration_results:
            if result["target_tpr"] == tpr and result["target_fpr"] == fpr:
                if result["gamma"] == gamma:
                    return result
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True) # 'opt', 'llama', 'mistral
    parser.add_argument('--algorithm', type=str, required=True) # 'KGW', 'SIR', 'EWD'
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--target_tpr', type=float, default=0.75)
    parser.add_argument('--target_fpr', type=float, default=0.01)
    parser.add_argument('--num_examples', type=int, default=200)
    parser.add_argument('--generation_length', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='calibration_results')
    args = parser.parse_args()

    print("Isolating delta for args:", args)

    already_computed = check_if_already_computed(args.model, args.algorithm, args.gamma, args.target_tpr, args.target_fpr, args.output_dir)
    if already_computed:
        print(f"Relevant record already exists at {os.path.join(args.output_dir, f'{args.model}_{args.algorithm}.json')}")
        print(json.dumps(already_computed, indent=2))
        print("Exiting...")
    else:
        model_and_tokenizer = ModelAndTokenizer(args.model)
        model = model_and_tokenizer.get_model()
        tokenizer = model_and_tokenizer.get_tokenizer()
        device = "cuda" if model_and_tokenizer.use_gpu else "cpu"

        delta = find_delta(
            model, tokenizer, device,
            args.algorithm, args.gamma, 
            args.target_tpr, args.target_fpr, 
            args.num_examples, args.generation_length
        )
        print(f'Found delta: {delta}')

        perplexity = assess_perplexity(
            model, tokenizer, device,
            args.algorithm, args.gamma, delta, 
            args.num_examples
        )
        print(f'Perplexity: {perplexity}')
        
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f'{args.model}_{args.algorithm}.json')
        print(f'Saving delta to {save_path}')
        add_evaluation_record_to_file(save_path, {
            "gamma": args.gamma,
            "target_tpr": args.target_tpr,
            "target_fpr": args.target_fpr,
            "delta": round(delta, 2),
            "perplexity": round(perplexity, 2)
        })

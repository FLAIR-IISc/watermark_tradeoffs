# ==================================================================
# dataset.py
# Description: Utility functions for loading and evaluating datasets
# ==================================================================

import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.utils import multiway_start_idx, bleu_score, f1_score, classification_score

dataset_name_to_path = {
    "sst2": "data/downstream/sst_requests.json",
    "boolq": "data/downstream/boolq_requests.json",
    "cb": "data/downstream/cb_requests.json",
    "hellaswag": "data/downstream/hellaswag_requests.json",
    "piqa": "data/downstream/piqa_requests.json",
    "drop": "data/downstream/drop_requests.json",
    "squad2": "data/downstream/squad2_requests.json",
    "wmt14enfr": "data/downstream/wmt14-en-fr_requests.json",
    "wmt20ende": "data/downstream/wmt20-en-de_requests.json"
}

def load_dataset(dataset_name, seed=151, num_examples=None):
    raw_dataset = json.load(open(dataset_name_to_path[dataset_name], 'r'))

    #### Classification Datasets ####
    if dataset_name == "sst2":
        dataset = []
        for i in range(len(raw_dataset)//2):
            prompt = raw_dataset[2*i][0]
            options = [raw_dataset[2*i][1], raw_dataset[2*i+1][1]]
            answer = raw_dataset[2*i][2]
            dataset.append((prompt, options, answer))
    elif dataset_name == "boolq":
        dataset = []
        for i in range(len(raw_dataset)//2):
            prompt = raw_dataset[2*i][0]
            options = [raw_dataset[2*i][1], raw_dataset[2*i+1][1]]
            answer = raw_dataset[2*i][2]
            dataset.append((prompt, options, answer))
    elif dataset_name == "cb":
        dataset = []
        for i in range(len(raw_dataset)//3):
            prompt = raw_dataset[3*i][0]
            options = [raw_dataset[3*i][1], raw_dataset[3*i+1][1], raw_dataset[3*i+2][1]]
            answer = raw_dataset[3*i][2]
            dataset.append((prompt, options, answer))

    #### Multiple Choice Datasets ####
    elif dataset_name in ["hellaswag"]:
        dataset = []
        for i in range(len(raw_dataset)//4):
            prompt = raw_dataset[4*i][0]
            options = [raw_dataset[4*i][1], raw_dataset[4*i+1][1], raw_dataset[4*i+2][1], raw_dataset[4*i+3][1]]
            answer = raw_dataset[4*i][2]
            dataset.append((prompt, options, answer))
    elif dataset_name == "piqa":
        dataset = []
        for i in range(len(raw_dataset)//2):
            prompt = raw_dataset[2*i][0]
            options = [raw_dataset[2*i][1], raw_dataset[2*i+1][1]]
            answer = raw_dataset[2*i][2]
            dataset.append((prompt, options, answer))
    
    #### Generation Datasets ####
    elif dataset_name in ["drop", "squad2", "wmt14enfr", "wmt20ende"]:
        dataset = raw_dataset
        
    else:
        raise ValueError(f"Invalid dataset '{dataset_name}'")
    
    # shuffle the dataset
    if seed is not None:
        random.seed(seed)
    random.shuffle(dataset)
    
    if num_examples is not None:
        dataset = dataset[:num_examples]
    
    return dataset

def evaluate_dataset(model, tokenizer, device, algorithm, gamma, delta, dataset, dataset_name):

    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=model.config.vocab_size,
        device=device,
        max_new_tokens=100,
        min_length=100 + 30,
        do_sample=False,
        no_repeat_ngram_size=4,
        stop_strings=["\n\n", "\n"]
    )
    watermark = AutoWatermark.load(
        algorithm, gamma, delta, 
        algorithm_config=os.path.join('config', f'{algorithm}.json'),
        transformers_config=transformers_config
    )    
    torch.manual_seed(0)

    outputs = []
    scores = []

    #### Classification Datasets ####
    if dataset_name in ["sst2", "boolq", "cb"]:
        for (prompt, options, answer) in tqdm(dataset, desc=f"Evaluating {dataset_name}"):
            options = [option.strip() for option in options]
            answer = answer.strip()
            prompt = prompt.strip()
            watermarked_logits = watermark.get_terminal_logits(prompt) # [vocab_size]
            option_logits = []
            for option in options:
                token_id = tokenizer.encode(prompt + " " + option)[-1]
                option_logit = watermarked_logits[token_id].item()
                option_logits.append(option_logit)
            answer_idx = options.index(answer)
            effective_gamma, effective_delta = watermark.get_effective_gamma_and_delta(prompt)

            score = classification_score(option_logits, answer_idx, effective_gamma, effective_delta)
            scores.append(score)
            outputs.append({
                'original_text': prompt,
                'options': options,
                'answer': answer,
                'score': score
            })
    
    #### Multiple Choice Datasets ####
    elif dataset_name in ["hellaswag", "piqa"]:        
        for (prompt, options, answer) in tqdm(dataset, desc=f"Evaluating {dataset_name}"):
            options = [option.strip() for option in options]
            answer = answer.strip()
            prompt = prompt.strip()
            ppls = []
            full_text_ids = [tokenizer.encode(prompt + " " + option, add_special_tokens=False) for option in options]
            start_idx = multiway_start_idx(full_text_ids)
            for option in options:
                full_text = prompt + " " + option
                ppls.append(watermark.get_perplexity(full_text, start_idx=start_idx))
            answer_idx = np.argmin(ppls)
            predicted_text = options[answer_idx].strip()
            answer = answer.strip()
            score = 1 if predicted_text == answer else 0
            scores.append(score)
            outputs.append({
                'original_text': prompt,
                'options': options,
                'predicted_text': predicted_text,
                'answer': answer,
                'score': score
            })

    #### Short Generation Datasets ####
    elif dataset_name in ["drop", "squad2"]:
        for (prompt, answer) in tqdm(dataset, desc=f"Evaluating {dataset_name}"):
            try:
                full_output_text = watermark.generate_watermarked_text(prompt)
                generated_text = full_output_text.split("Answer: ")[-1].strip()
                answer = answer.strip()
                score = f1_score(answer, generated_text)
                scores.append(score)
                outputs.append({
                    'original_text': prompt,
                    'generated_text': generated_text,
                    'answer': answer,
                    'score': score
                })
            except:
                print(f"Error for prompt: \"{prompt}\". Skipping...")
                continue
    
    #### Long Generation Datasets ####
    elif dataset_name in ["wmt14enfr", "wmt20ende"]:
        for (prompt, answer) in tqdm(dataset, desc=f"Evaluating {dataset_name}"):
            full_output_text = watermark.generate_watermarked_text(prompt)
            generated_text = full_output_text.split(" phrase:")[-1].strip()
            answer = answer.strip()
            score = bleu_score(answer, generated_text)
            scores.append(score)
            outputs.append({
                'original_text': prompt,
                'generated_text': generated_text,
                'answer': answer,
                'score': score
            })
    else:
        raise NotImplementedError

    return np.mean(scores), outputs

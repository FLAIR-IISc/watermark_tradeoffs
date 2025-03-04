# ========================================================
# utils.py
# Description: Utility functions for downstream evaluation
# ========================================================

import itertools
import numpy as np

import nltk; nltk.download('punkt'); nltk.download('punkt_tab')

#### Generation Parsing Functions ####
def two_way_start_idx(list1, list2):
    start_idx = 0
    for i in range(1, min(len(list1), len(list2))):
        if list1[i] != list2[i]:
            break
        start_idx = i
    return start_idx

def multiway_start_idx(list_of_lists):
    start_idx = float('inf')
    for i in range(len(list_of_lists) - 1):
        for j in range(i+1, len(list_of_lists)):
            new_start_idx = two_way_start_idx(list_of_lists[i], list_of_lists[j])
            start_idx = min(start_idx, new_start_idx)
    return start_idx



#### Evaluation Metrics ####
def bleu_score(answer, generated_text):
        return nltk.translate.bleu_score.sentence_bleu([answer], generated_text)

def f1_score(answer, generated_text):
    answer_tokens = set(nltk.word_tokenize(answer))
    generated_tokens = set(nltk.word_tokenize(generated_text))
    if len(answer_tokens) == 0 or len(generated_tokens) == 0:
        return 0
    precision = len(answer_tokens.intersection(generated_tokens)) / len(generated_tokens)
    recall = len(answer_tokens.intersection(generated_tokens)) / len(answer_tokens)
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

def classification_score(option_logits, answer_idx, gamma, delta):
    mask_length = len(option_logits)
    option_logits = np.array(option_logits)

    expected_score = 0
    # create all possible masks
    for mask in itertools.product([0, 1], repeat=mask_length):
        mask = np.array(mask)
        watermarked_logits = option_logits + delta * mask
        predicted_idx = np.argmax(watermarked_logits)
        if predicted_idx == answer_idx:
            expected_score += (gamma ** np.sum(mask)) * ((1-gamma) ** (mask_length - np.sum(mask)))
    return expected_score
# ================================================================
# model_loading.py
# Description: Utility functions for loading models and tokenizers
# ================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_to_path = {
    'opt': 'facebook/opt-6.7b',
    'llama': 'huggyllama/llama-7b',
    'mistral': 'mistralai/Mistral-7B-v0.1'
}

class ModelAndTokenizer:
    def __init__(self, model_name, use_gpu=None):
        self.model_name = model_name
        self.model_path = model_name_to_path[model_name]
        
        self.model = None
        self.tokenizer = None
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
    
    def get_model(self):
        if self.model is None:
            if self.use_gpu:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="cuda").to(torch.bfloat16).eval()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(torch.float16).eval()
        return self.model
    
    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=(not "opt" in self.model_name))
            if "mistral" in self.model_name:
                self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        return self.tokenizer

# ============================================
# sir.py
# Description: Implementation of SIR algorithm
# ============================================

import os
import json
import jieba
import torch
import numpy as np
from functools import partial
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList, BertTokenizer, BertModel

from ..base import BaseWatermark
from .transform_model import TransformModel
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from utils.utils import load_config_file


class SIRConfig:
    """Config class for SIR algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, gamma, delta, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the SIR configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/SIR.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'SIR':
            raise AlgorithmNameMismatchError('SIR', config_dict['algorithm_name'])

        self.delta = config_dict['delta'] if delta is None else delta
        self.chunk_length = config_dict['chunk_length']
        self.scale_dimension = config_dict['scale_dimension']
        self.z_threshold = config_dict['z_threshold']
        self.transform_model_input_dim = config_dict['transform_model_input_dim']
        self.transform_model_name = config_dict['transform_model_name']
        self.embedding_model_path = config_dict['embedding_model_path']
        self.mapping_dir = config_dict['mapping_dir']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class SIRUtils:
    """Utility class for SIR algorithm, contains helper functions."""

    def __init__(self, config: SIRConfig, *args, **kwargs) -> None:
        """
            Initialize the SIR utility class.

            Parameters:
                config (SIRConfig): Configuration for the SIR algorithm.
        """
        self.config = config
        self.transform_model = self._get_transform_model(self.config.transform_model_name, config.transform_model_input_dim).to(self.config.device)
        self.embedding_tokenizer = BertTokenizer.from_pretrained(self.config.embedding_model_path)
        self.embedding_model = BertModel.from_pretrained(self.config.embedding_model_path).to(self.config.device)
        self.mapping = self._get_mapping(self.config.mapping_dir)

    def get_embedding(self, sentence: str) -> torch.FloatTensor:
        """Get the embedding of the input sentence."""
        input_ids = self.embedding_tokenizer.encode(sentence, return_tensors="pt", max_length=512, truncation="longest_first")
        input_ids = input_ids.to(self.config.device)
        with torch.no_grad():
            output = self.embedding_model(input_ids)
        return output[0][:, 0, :]
    
    def get_text_split(self, sentence: str) -> list[list[str]]:
        """Split the input text into chunks of words."""
        words = list(jieba.cut(sentence))
        non_space_indices = [index for index, word in enumerate(words) if word.strip()]
        words_2d = []
        chunk_start = 0
        for i in range(0, len(non_space_indices), self.config.chunk_length):
            chunk_end = i + self.config.chunk_length
            chunk_end = min(chunk_end, len(non_space_indices))
            chunk_indices = non_space_indices[:chunk_end]
            if chunk_indices:
                chunk = words[chunk_start:chunk_indices[-1] + 1]
                words_2d.append(chunk)
            chunk_start = chunk_indices[-1] + 1
        return words_2d

    def scale_vector(self, v: np.array) -> np.array:
        """Scale the input vector using tanh function."""
        mean = np.mean(v)
        v_minus_mean = v - mean
        v_minus_mean = np.tanh(1000 * v_minus_mean)
        return v_minus_mean
    
    def _get_mapping(self, mapping_dir: str) -> list[int]:
        """Get the mapping for the input tokens."""
        input_size = self.config.vocab_size
        mapping_name = os.path.join(mapping_dir, f'300_mapping_{input_size}.json')
        print(f"Looking for mapping at {mapping_name}")

        # try loading mapping from the provided mapping path
        try:
            with open(mapping_name, 'r') as f:
                mapping = json.load(f)

        # if the file does not exist, create a new mapping and save it to the provided mapping path
        except:
            raise FileNotFoundError(f"Mapping file not found at {mapping_name}")
        return mapping
    
    def _get_context_sentence(self, input_ids: torch.LongTensor):
        """Get the context sentence from the input_ids."""
        sentence = self.config.generation_tokenizer.decode(input_ids, skip_special_tokens=True)
        words_2d = self.get_text_split(sentence)
        if len(words_2d) == 0:
            return ''
        if len(words_2d[-1]) == self.config.chunk_length:
            return ''.join([''.join(group) for group in words_2d]).strip()
        else:
            return ''.join([''.join(group) for group in words_2d[:-1]]).strip()
    
    def _get_transform_model(self, model_name: str, input_dim: int) -> TransformModel:
        """Get the transform model from the provided model name."""
        model = TransformModel(input_dim=input_dim)
        model.load_state_dict(torch.load(model_name))
        return model
    
    def get_bias(self, input_ids: torch.LongTensor) -> list[int]:
        """Get the bias for the input_ids."""
        with torch.no_grad():
            context_sentence = self._get_context_sentence(input_ids)
            context_embedding = self.get_embedding(context_sentence)
            output = self.transform_model(context_embedding).cpu()[0].numpy()
        similarity_array = self.scale_vector(output)[self.mapping]
        return -similarity_array
    
    def _get_effective_gamma_and_delta(self, input_ids: torch.LongTensor) -> tuple[float, float]:
        """Get the effective gamma for the input_ids."""
        with torch.no_grad():
            context_sentence = self._get_context_sentence(input_ids)
            context_embedding = self.get_embedding(context_sentence)
            output = self.transform_model(context_embedding).cpu()[0].numpy()
        similarity_array = self.scale_vector(output)
        bias = -similarity_array
        effective_gamma = np.mean(bias/2 + 0.5) # shift [-1, 1] bias to the more standard [0, 1] bias
        effective_delta = 2 * self.config.delta # double delta to account for halving range of bias
        # Now, biasing logits by effective_gamma and effective_delta should be equivalent to biasing by the original bias in terms
        # of ordinal relationship among logits, and probability distribution of the output (since softmax is invariant to constant shifts)
        return effective_gamma, effective_delta

class SIRLogitsProcessor(LogitsProcessor):
    """Logits processor for SIR algorithm."""

    def __init__(self, config: SIRConfig, utils: SIRUtils, *args, **kwargs):
        """
            Initialize the SIR logits processor.

            Parameters:
                config (SIRConfig): Configuration for the SIR algorithm.
                utils (SIRUtils): Utility class for the SIR algorithm.
        """
        self.config = config
        self.utils = utils
    
    def _bias_logits(self, scores: torch.LongTensor, batched_bias: torch.FloatTensor) -> torch.FloatTensor:
        """Bias the logits using the batched_bias."""
        scores = scores + batched_bias * self.config.delta
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process the logits to add watermark."""
        batched_bias = [None for _ in range(input_ids.shape[0])] 

        for b_idx in range(input_ids.shape[0]):
            current_bias = self.utils.get_bias(input_ids[b_idx])
            batched_bias[b_idx] = current_bias

        batched_bias_np = np.array(batched_bias)
        batched_bias = torch.Tensor(batched_bias_np).to(scores.device)

        scores = self._bias_logits(scores=scores, batched_bias=batched_bias)
        return scores


class SIR(BaseWatermark):
    """Top-level class for SIR algorithm."""

    def __init__(self, algorithm_config: str, gamma, delta, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the SIR algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = SIRConfig(algorithm_config, gamma, delta, transformers_config)
        self.utils = SIRUtils(self.config)
        self.logits_processor = SIRLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            tokenizer=self.config.generation_tokenizer,
            **self.config.gen_kwargs
        )
        
        # encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the input text."""

        # Split the text into a 2D array of words
        word_2d = self.utils.get_text_split(text)

        # Initialize a list to hold all computed values for similarity
        all_value = []

        # Iterate over each sentence in the split text, skipping the first
        for i in range(1, len(word_2d)):
            # Create the context sentence from all previous text portions
            context_sentence = ''.join([''.join(group) for group in word_2d[:i]]).strip()
            # Current sentence to check against the context
            current_sentence = ''.join(word_2d[i]).strip()

            # Continue if the context sentence is shorter than the required chunk length
            if len(list(jieba.cut(context_sentence))) < self.config.chunk_length:
                continue

            # Get embedding of the context sentence
            context_embedding = self.utils.get_embedding(context_sentence)
            # Transform the embedding using the model, process output
            output = self.utils.transform_model(context_embedding).cpu().detach()[0].numpy()
            # Scale the output vector and map to predefined indices
            similarity_array = self.utils.scale_vector(output)[self.utils.mapping]

            # Encode the current sentence into tokens
            tokens = self.config.generation_tokenizer.encode(current_sentence, return_tensors="pt", add_special_tokens=False)

            # Append negative similarity values for each token in the current sentence
            for index in tokens[0]:
                all_value.append(-float(similarity_array[index]))

        # Calculate the mean of all similarity values
        z_score = np.mean(all_value)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)
    
    def get_perplexity(self, text: str, start_idx=1, *args, **kwargs) -> float:
        """Get perplexity of the text under the watermarked model."""
        
        # Encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt").to(self.config.device)
        input_ids = encoded_text["input_ids"][0] # [seq_len]
        with torch.no_grad():
            logits = self.config.generation_model(**encoded_text).logits[0] # [seq_len, vocab_size]
            new_logits = torch.zeros_like(logits) # [seq_len, vocab_size]
            for t_idx in range(start_idx, input_ids.shape[0]):
                new_logits[t_idx-1, :] = self.logits_processor(input_ids[:t_idx].unsqueeze(0), logits[t_idx-1, :].unsqueeze(0)).squeeze(0)
        
        # Compute perplexity
        perplexity = F.cross_entropy(new_logits[start_idx-1:-1, :], input_ids[start_idx:], reduction='mean').exp().item()
        
        return perplexity

    def get_terminal_logits(self, text: str, *args, **kwargs) -> torch.Tensor:
        """Get terminal logits for the text."""
        
        # Encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.config.device)
        input_ids = encoded_text["input_ids"][0] # [seq_len]
        with torch.no_grad():
            logits = self.config.generation_model(**encoded_text).logits[0] # [seq_len, vocab_size]
            t_idx = input_ids.shape[0] - 1
            terminal_logits = self.logits_processor(input_ids[:t_idx].unsqueeze(0), logits[t_idx, :].unsqueeze(0)).squeeze(0) # [vocab_size]
        return terminal_logits.cpu()
    
    def get_effective_gamma_and_delta(self, text: str, *args, **kwargs) -> tuple[float, float]:
        """Get effective gamma and delta for the text."""
        
        # Encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.config.device)
        input_ids = encoded_text["input_ids"][0] # [seq_len]
        effective_gamma, effective_delta = self.utils._get_effective_gamma_and_delta(input_ids)
        return effective_gamma, effective_delta
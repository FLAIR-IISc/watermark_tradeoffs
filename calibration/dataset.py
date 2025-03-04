# ===========================================
# dataset.py
# Description: Dataset classes for evaluation
# ===========================================

import json


class BaseDataset:
    """Base class for dataset."""

    def __init__(self):
        """Initialize the dataset."""
        self.prompts = []
        self.natural_texts = []
        self.references = []

    @property
    def prompt_nums(self):
        """Return the number of prompts."""
        return len(self.prompts)

    @property
    def natural_text_nums(self):
        """Return the number of natural texts."""
        return len(self.natural_texts)

    @property
    def reference_nums(self):
        """Return the number of references."""
        return len(self.references)

    def get_prompt(self, index):
        """Return the prompt at the specified index."""
        return self.prompts[index]

    def get_natural_text(self, index):
        """Return the natural text at the specified index."""
        return self.natural_texts[index]

    def get_reference(self, index):
        """Return the reference at the specified index."""
        return self.references[index]

    def load_data(self):
        """Load and process data to populate prompts, natural_texts, and references."""
        pass

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.prompts)

class C4Dataset(BaseDataset):
    """Dataset class for C4 dataset."""

    def __init__(self, data_source: str, num_examples: int = 200) -> None:
        """
            Initialize the C4 dataset.

            Parameters:
                data_source (str): The path to the C4 dataset file.
        """
        super().__init__()
        self.data_source = data_source
        self.num_examples = num_examples
        self.load_data()
    
    def load_data(self):
        """Load data from the C4 dataset file."""
        with open(self.data_source, 'r') as f:
           lines = f.readlines()
        for line in lines[:self.num_examples]:
            item = json.loads(line)
            self.prompts.append(item['prompt'])
            self.natural_texts.append(item['natural_text'])

if __name__ == '__main__':
    d1 = C4Dataset('data/calibration/processed_c4.json')

from typing import List, Tuple
import numpy as np
import collections
from torch.utils import data


import IPython

np.random.seed(42)


def generate_dataset(n_sequences: int = 100) -> List[List[str]]:
    """Generate sample dataset.

    Samples have a changing (between samples) but fixed number (within sample) of a's
    followed by b's followed by an EOS token.

    Args:
        n_sequences (int): Number of data samples to generate.

    Returns:
        samples (List): List of data samples.
    """
    samples = []
    for _ in range(n_sequences):
        num_tokens = np.random.randint(1, 10)
        sample = ["a"] * num_tokens + ["b"] * num_tokens + ["EOS"]
        samples.append(sample)
    return samples


def sequences_to_dict(
    sequences: List[List[str]],
) -> Tuple[collections.defaultdict, collections.defaultdict, int, int]:
    """Generate vocabulary dictionaries for the given sequence.

    Contains all unique words, the EOS and UNK token.

    Args:
        sequences (List[List[str]]): List of data samples.

    Returns:
        word_to_idx (collections.defaultdict): Dictionary mapping words to indicies.
        idx_to_word (collections.defaultdict): Dictionary mapping indicies to words.
        num_sequences (int): Number of sequences.
        vocab_size (collections.defaultdict): Size of vocabulary (unique words).
    """
    all_words = [item for sublist in sequences for item in sublist]
    word_count = collections.defaultdict(int)
    for word in all_words:
        word_count[word] += 1

    unique_words = [word for word in word_count]
    unique_words.append("UNK")

    num_sequences, vocab_size = len(sequences), len(unique_words)

    word_to_idx = collections.defaultdict()
    idx_to_word = collections.defaultdict()
    for idx, word in enumerate(unique_words):
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    return word_to_idx, idx_to_word, num_sequences, vocab_size


class Dataset(data.Dataset):
    """Define Dataset class
    """
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]
        return X, y


def _get_inputs_targets_from_sequences(sequences):
    inputs, targets = [], []
    for sequence in sequences:
        inputs.append(sequence[:-1])
        targets.append(sequence[1:])
    return inputs, targets
    
def create_datasets(sequences: List[List[str]], dataset_class: Dataset, p_train: int = 0.8, p_val: int = 0.1, p_test: int = 0.1):
    """Create split dataset from input sequences.

    Args:
        sequences (List[List[str]]): Data sequences.
        dataset_class (Dataset): Dataset class structure.
        p_train (int): Percentage of dataset split to train.
        p_val (int): Percentage of dataset split to validation.
        p_test (int): Percentage of dataset split to test.
    
    Returns:
        training_set (Dataset): Training partitioned dataset.
        validation_set (Dataset): Validation partitioned dataset.
        test_set (Dataset): Test partitioned dataset.
    
    """
    num_train = int(len(sequences)*p_train)
    num_val = int(len(sequences)*p_val)
    num_test = int(len(sequences)*p_test)

    sequences_train = sequences[:num_train]
    sequences_val = sequences[num_train:num_train+num_val]
    sequences_test = sequences[-num_test:]

    inputs_train, targets_train = _get_inputs_targets_from_sequences(sequences_train)
    inputs_val, targets_val = _get_inputs_targets_from_sequences(sequences_val)
    inputs_test, targets_test = _get_inputs_targets_from_sequences(sequences_test)

    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_val, targets_val)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, validation_set, test_set

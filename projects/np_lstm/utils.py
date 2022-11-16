from typing import List, Tuple
import numpy as np
import collections
import datasets

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

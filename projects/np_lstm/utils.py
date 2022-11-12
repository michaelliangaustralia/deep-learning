from typing import List
import numpy as np
import collections

import IPython
np.random.seed(42)

def generate_dataset(n_sequences: int = 100) -> List[List[str]]:
    """Generate sample dataset.
    
    Samples have a changing (between samples) but fixed number (within sample) of a's followed by b's followed by an EOS token

    Args:
        n_sequences (int): Number of data samples to generate. 
    
    Returns:
        samples (List): List of data samples
    """
    samples = []
    for _ in range(n_sequences):
        num_tokens = np.random.randint(1, 10)
        sample = ['a'] * num_tokens + ['b'] * num_tokens + ['EOS']
        samples.append(sample)
    return samples

def sequences_to_dict(sequences: List[List[str]]):
    """Generate vocabulary dictionaries for the given sequence.

    Contains all unique words, the EOS and UNK token.

    Args:
        sequences (List[List[str]]): List of data samples
    
    Returns:
        TODO
    """
    all_words = [item for sublist in sequences for item in sublist]
    word_count = collections.defaultdict(int)
    for word in all_words:
        word_count[word] += 1

    unique_words = [word for word in word_count]
    unique_words.append('UNK')

    num_sentences, vocab_size = len(sequences), len(unique_words)

    # TODO continue here from https://github.com/CaptainE/RNN-LSTM-in-numpy/blob/master/RNN_LSTM_from_scratch.ipynb
    
    IPython.embed()
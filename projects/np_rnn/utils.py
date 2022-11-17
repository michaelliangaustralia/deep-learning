from typing import List, Tuple, Dict
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

def one_hot_encode(idx: int, vocab_size: int) -> np.ndarray:
    """One-hot encodes a single word given its index and the size of the vocabulary.
    
    Args:
        idx (int): the index of the given word
        vocab_size (int): the size of the vocabulary
    
    Returns:
        one_hot (np.ndarray): A zero'd 1-D numpy array of length vocab_size with value
        1.0 at the given index.
    """
    one_hot = np.zeros(vocab_size)
    one_hot[idx] = 1.0
    return one_hot


def one_hot_encode_sequence(sequence: List, vocab_size: int, word_to_idx: Dict) -> np.ndarray:
    """One-hot encodes a sequence of words given a fixed vocabulary size.
    
    Args:
        sequence (List): List of words to encode.
        vocab_size (int): Size of the vocabulary.
        word_to_idx (Dict): Dictionary mapping words to indices.
     
    Returns:
        encoding (np.ndarray): a 3-D numpy array of shape (num words, vocab size, 1).
    """
    encoding = np.array([one_hot_encode(word_to_idx[word], vocab_size) for word in sequence])
    encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1)
    return encoding

def _init_orthogonal(param):
    """Initializes weight parameters orthogonally.
    
    Refer to this paper for an explanation of this initialization:
    https://arxiv.org/abs/1312.6120.
    """
    if param.ndim < 2:
        raise ValueError("Only parameters with 2 or more dimensions are supported.")
    rows, cols = param.shape
    new_param = np.random.randn(rows, cols)
    if rows < cols:
        new_param = new_param.T
    q, r = np.linalg.qr(new_param)
    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph
    if rows < cols:
        q = q.T
    new_param = q
    return new_param

def init_rnn(hidden_size: int, vocab_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize the RNN.

    Args:
        hidden_size (int): Size of the hidden layer.
        vocab_size (int): Size of the hidden layer.

    Returns:
        weight_matrix_input (np.ndarray): weight matrix of the input-rnn interface.
        weight_matrix_rnn (np.ndarray): weight matrix of the rnn-rnn interface.
        weight_matrix_output (np.ndarray): weight matrix of the rnn-output interface.
        b_hidden (np.ndarray): bias matrix of the rnn.
        b_out (np.ndarray): bias matrix of the rnn-output interface.
    """
    weight_matrix_input = np.zeros((hidden_size, vocab_size))
    weight_matrix_rnn = np.zeros((hidden_size, hidden_size))
    weight_matrix_output = np.zeros((vocab_size, hidden_size))
    # weight_matrix_output = np.zeros((hidden_size, vocab_size))

    b_hidden = np.zeros((hidden_size, 1))
    b_out = np.zeros((vocab_size, 1))

    weight_matrix_input = _init_orthogonal(weight_matrix_input)
    weight_matrix_rnn = _init_orthogonal(weight_matrix_rnn)
    weight_matrix_output = _init_orthogonal(weight_matrix_output)

    return weight_matrix_input, weight_matrix_rnn, weight_matrix_output, b_hidden, b_out

def sigmoid(x, derivative: bool = False):
    """Sigmoid activation function.

    Args:
        x: Input values.
        derivative (bool): Will return derivative if True.

    Returns:
        f: Input with sigmoid activation function applied over it.
    """
    x_safe = x + 1e-12
    f = 1 / (1 + np.exp(-x_safe))
    if derivative:
        return f * (1 - f)
    else:
        return f

def tanh(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Tanh activation function.

    Args:
        x (np.ndarray): The array where the function is applied.
        derivative: If set to True will return the derivative instead of the forward pass.

    Return:
        f (np.ndarray): Array with tanh function computed over it.
    """
    x_safe = x + 1e-12
    f = (np.exp(x_safe)-np.exp(-x_safe))/(np.exp(x_safe)+np.exp(-x_safe))
    
    if derivative: # Return the derivative of the function evaluated at x
        return 1-f**2
    else: # Return the forward pass of the function at x
        return f

def softmax(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Calculate the softmax of an array x.

    Args:
        x (np.ndarray): The array where the function is applied.
        derivative (bool): If set to True will return the derivative instead of the forward pass.

    Return:
        f (np.ndarray): Softmaxed array.
    """
    x_safe = x + 1e-12
    f = np.exp(x_safe) / np.sum(np.exp(x_safe))

    if derivative:
        pass
    else:
        return f

def forward_pass(inputs: np.ndarray, hidden_state: np.ndarray, params: Tuple[Dict, Dict, Dict, Dict, Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the forward pass of a vanilaa RNN.
    
    Args:
        inputs (np.ndarray): Sequence of inputs to be processed,
        hidden_state (np.ndarray): Previous hidden state of the RNN.
        params (Tuple[Dict, Dict, Dict, Dict, Dict]): Parameters of the RNN.

    Returns:
        outputs (np.ndarray): List of outputs of the RNN.
        hidden_states (np.ndarray): List of RNN hidden states for each input.
    """
    weight_matrix_input, weight_matrix_rnn, weight_matrix_output, b_hidden, b_out = params
    outputs, hidden_states = [], []
    for inp in inputs:
        IPython.embed()
        hidden_state = tanh(np.dot(weight_matrix_input, inp) + np.dot(weight_matrix_rnn, hidden_state) + b_hidden)
        output = softmax(np.dot(weight_matrix_output, hidden_state) + b_out)
        outputs.append(output)
        hidden_states.append(hidden_state)
    return outputs, hidden_states


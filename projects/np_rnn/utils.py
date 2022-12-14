from typing import List, Tuple, Dict
import numpy as np
import collections
from torch.utils import data


import IPython

np.random.seed(42)


def generate_dataset(num_sequences: int = 100) -> List[List[str]]:
    """Generate sample dataset.

    Samples have a changing (between samples) but fixed number (within sample) of a's
    followed by b's followed by an EOS token.

    Args:
        num_sequences (int): Number of data samples to generate.

    Returns:
        samples (List): List of data samples.
    """
    samples = []
    for _ in range(num_sequences):
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
    """Define Dataset class"""

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]
        return X, y


def _get_inputs_targets_from_sequences(sequences: np.ndarray) -> np.ndarray:
    inputs, targets = [], []
    for sequence in sequences:
        inputs.append(sequence[:-1])
        targets.append(sequence[1:])
    return inputs, targets


def create_datasets(
    sequences: List[List[str]],
    dataset_class: Dataset,
    p_train: int = 0.8,
    p_val: int = 0.1,
    p_test: int = 0.1,
):
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
    num_train = int(len(sequences) * p_train)
    num_val = int(len(sequences) * p_val)
    num_test = int(len(sequences) * p_test)

    sequences_train = sequences[:num_train]
    sequences_val = sequences[num_train : num_train + num_val]
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


def one_hot_encode_sequence(
    sequence: List, vocab_size: int, word_to_idx: Dict
) -> np.ndarray:
    """One-hot encodes a sequence of words given a fixed vocabulary size.

    Args:
        sequence (List): List of words to encode.
        vocab_size (int): Size of the vocabulary.
        word_to_idx (Dict): Dictionary mapping words to indices.

    Returns:
        encoding (np.ndarray): a 3-D numpy array of shape (num words, vocab size, 1).
    """
    encoding = np.array(
        [one_hot_encode(word_to_idx[word], vocab_size) for word in sequence]
    )
    encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1)
    return encoding


def _init_orthogonal(param: np.ndarray) -> np.ndarray:
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


def init_rnn(
    hidden_size: int, vocab_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize the RNN.

    Args:
        hidden_size (int): Size of the hidden layer.
        vocab_size (int): Size of the hidden layer.

    Returns:
        w_input (np.ndarray): weight matrix of the input-rnn interface.
        w_rnn (np.ndarray): weight matrix of the rnn-rnn interface.
        w_output (np.ndarray): weight matrix of the rnn-output interface.
        b_hidden (np.ndarray): bias matrix of the rnn.
        b_out (np.ndarray): bias matrix of the rnn-output interface.
    """
    w_input = np.zeros((hidden_size, vocab_size))
    w_rnn = np.zeros((hidden_size, hidden_size))
    w_output = np.zeros((vocab_size, hidden_size))

    b_hidden = np.zeros((hidden_size, 1))
    b_out = np.zeros((vocab_size, 1))

    w_input = _init_orthogonal(w_input)
    w_rnn = _init_orthogonal(w_rnn)
    w_output = _init_orthogonal(w_output)

    return w_input, w_rnn, w_output, b_hidden, b_out


def sigmoid(x: np.ndarray, derivative: bool = False) -> np.ndarray:
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
    f = (np.exp(x_safe) - np.exp(-x_safe)) / (np.exp(x_safe) + np.exp(-x_safe))

    if derivative:  # Return the derivative of the function evaluated at x
        return 1 - f**2
    else:  # Return the forward pass of the function at x
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


def forward_pass_rnn(
    inputs: np.ndarray,
    hidden_state: np.ndarray,
    params: Tuple[Dict, Dict, Dict, Dict, Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the forward pass of a vanilla RNN.

    Args:
        inputs (np.ndarray): Sequence of inputs to be processed,
        hidden_state (np.ndarray): Previous hidden state of the RNN.
        params (Tuple[Dict, Dict, Dict, Dict, Dict]): Parameters of the RNN.

    Returns:
        outputs (np.ndarray): List of outputs of the RNN.
        hidden_states (np.ndarray): List of RNN hidden states for each input.
    """
    (
        w_input,
        w_rnn,
        w_output,
        b_hidden,
        b_out,
    ) = params
    outputs, hidden_states = [], []
    for inp in inputs:
        hidden_state = tanh(
            np.dot(w_input, inp) + np.dot(w_rnn, hidden_state) + b_hidden
        )
        output = softmax(np.dot(w_output, hidden_state) + b_out)
        outputs.append(output)
        hidden_states.append(hidden_state.copy())
    return outputs, hidden_states


def clip_gradient_norm(grads: np.ndarray, max_norm: float = 0.25) -> np.ndarray:
    """Clip gradients to avoid exploding gradients problem."""
    max_norm = float(max_norm)
    total_norm = 0

    for grad in grads:
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm

    total_norm = np.sqrt(total_norm)

    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for grad in grads:
            grad *= clip_coef

    return grads


def backward_pass_rnn(
    inputs: np.ndarray,
    outputs: np.ndarray,
    hidden_states: np.ndarray,
    targets: np.ndarray,
    params: np.ndarray,
) -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Computes the backward pass of a vanilla RNN.

    Compute gradient by -1 to the softmaxed probability of the output for the target index.

    Args:
        inputs (np.ndarray): RNN inputs.
        outputs (np.ndarray): RNN outputs.
        hidden_states (np.ndarray): RNN hidden state.
        targets (np.ndarray): Sample targets.
        params (np.ndarray): RNN state.

    Returns:
        loss (float): Loss computation for the set of outputs.
        grads (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): Gradient matrices
            for weight matrices and biases.
    """

    (
        w_input,
        w_rnn,
        w_output,
        b_hidden,
        b_out,
    ) = params

    d_w_input, d_w_rnn, d_w_output = (
        np.zeros_like(w_input),
        np.zeros_like(w_rnn),
        np.zeros_like(w_output),
    )
    d_b_hidden, d_b_out = np.zeros_like(b_hidden), np.zeros_like(b_out)

    d_hidden_next = np.zeros_like(hidden_states[0])
    loss = 0

    for t in reversed(range(len(outputs))):
        loss += -np.mean(np.log(outputs[t] + 1e-12) * targets[t])

        d_output = outputs[t].copy()
        d_output[np.argmax(targets[t])] -= 1

        d_w_output += np.dot(d_output, hidden_states[t].T)
        d_b_out += d_output

        d_hidden = np.dot(w_output.T, d_output) + d_hidden_next

        d_f = tanh(hidden_states[t], derivative=True) * d_hidden  # ?
        d_b_hidden += d_f

        d_w_input += np.dot(d_f, inputs[t].T)

        d_w_rnn += np.dot(d_f, hidden_states[t - 1].T)
        d_hidden_next = np.dot(w_rnn.T, d_f)

    grads = (
        d_w_input,
        d_w_rnn,
        d_w_output,
        d_b_hidden,
        d_b_out,
    )

    grads = clip_gradient_norm(grads)

    return loss, grads


def update_parameters(
    params: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    grads: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lr: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update parameters with gradient descent.

    Args:
        params (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]): State of RNN.
        grads (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]): Gradient matrix of RNN.
        lr (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]): Learning rate.

    Returns:
        grads (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]): Updated state of RNN.
    """
    for param, grad in zip(params, grads):
        param -= lr * grad
    return params


def inference(
    params: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    vocab_size: int,
    hidden_size: int,
    idx_to_word: collections.defaultdict,
    word_to_idx: collections.defaultdict,
    sentence: str = "",
    num_generate: int = 4,
):
    """Runs inference over an input sentence.

    Args:
        params (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]): Network weights.
        sentence (str): String with whitespace-separated tokens.
        vocab_size (str): Size of vocabulary.
        hidden_size (str): Size of hidden state.
        num_generate (int): The number of tokens to generate.
    """
    sentence = sentence.split(" ")

    sentence_one_hot = one_hot_encode_sequence(sentence, vocab_size, word_to_idx)

    # Initialize hidden state as zeros
    hidden_state = np.zeros((hidden_size, 1))

    # Generate hidden state for sentence
    outputs, hidden_states = forward_pass_rnn(sentence_one_hot, hidden_state, params)

    # Output sentence
    output_sentence = sentence

    # Append first prediction
    word = idx_to_word[np.argmax(outputs[-1])]
    output_sentence.append(word)

    # Forward pass
    for i in range(num_generate):

        # Get the latest prediction and latest hidden state
        output = outputs[-1]
        hidden_state = hidden_states[-1]

        # Reshape our output to match the input shape of our forward pass
        output = output.reshape(1, output.shape[0], output.shape[1])

        # Forward pass
        outputs, hidden_states = forward_pass_rnn(output, hidden_state, params)

        # Compute the index the most likely word and look up the corresponding word
        word = idx_to_word[np.argmax(outputs)]

        output_sentence.append(word)

    return output_sentence


def init_lstm(
    hidden_size: int, vocab_size: int, z_size: int
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Initialises our LSTM network.

    Args:
        hidden_size (int): Size of hidden state.
        vocab_size (int): Size of vocabulary.
        z_size (int): Size of combined hidden state and vocabulary.

    Returns:
        w_forget (np.ndarray): Weight matrix of the forget gate.
        w_input (np.ndarray): Weight matrix of the input gate.
        w_candidate (np.ndarray): Weight matrix of the candidate gate.
        w_output (np.ndarray): Weight matrix of the output gate.
        w_logit (np.ndarray): Weight matrix relating the hidden state to the output for particular timestep.
        b_forget (np.ndarray): Bias matrix of the forget gate.
        b_input (np.ndarray): Bias matrix of the b_input gate.
        b_candidate (np.ndarray): Bias matrix of the candidate gate.
        b_output (np.ndarray): Bias matrix of the output gate.
        b_v (np.ndarray): Bias matrix relating the hidden state to the output.
    """
    w_forget = np.random.randn(hidden_size, z_size)
    b_forget = np.zeros((hidden_size, 1))

    w_input = np.random.randn(hidden_size, z_size)
    b_input = np.zeros((hidden_size, 1))

    w_candidate = np.random.randn(hidden_size, z_size)
    b_candidate = np.zeros((hidden_size, 1))

    w_output = np.random.randn(hidden_size, z_size)
    b_output = np.zeros((hidden_size, 1))

    w_logit = np.random.randn(vocab_size, hidden_size)
    b_logit = np.zeros((vocab_size, 1))

    w_forget = _init_orthogonal(w_forget)
    w_input = _init_orthogonal(w_input)
    w_candidate = _init_orthogonal(w_candidate)
    w_output = _init_orthogonal(w_output)
    w_logit = _init_orthogonal(w_logit)

    return (
        w_forget,
        w_input,
        w_candidate,
        w_output,
        w_logit,
        b_forget,
        b_input,
        b_candidate,
        b_output,
        b_logit,
    )


def forward_pass_lstm(
    inputs: np.ndarray,
    hidden_prev: np.ndarray,
    candidate_prev: np.ndarray,
    params: np.ndarray,
    hidden_size: int,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
]:
    """Forward pass of the LSTM network.

    Args:
        inputs (np.ndarray): Inputs.
        hidden_prev (np.ndarray): Previous hidden state.
        candidate_prev (np.ndarray): Previous candidate state.
        params (np.ndarray): List of LSTM params.
        hidden_size (int): Size of LSTM hidden states.

    Returns:
        z_s (List[np.ndarray]): List of prev hidden state and input row stack outputs.
        f_s (List[np.ndarray]): List of forget gate outputs.
        i_s (List[np.ndarray]): List of input gate outputs.
        g_s (List[np.ndarray]): List of candidate outputs.
        C_s (List[np.ndarray]): List of memory cell outputs.
        o_s (List[np.ndarray]): List of output gate outputs.
        h_s (List[np.ndarray]): List of hidden state outputs.
        v_s (List[np.ndarray]): List of logit outputs.
        output_s (List[np.ndarray]): List of softmaxed logit outputs.
    """
    assert hidden_prev.shape == (hidden_size, 1)
    assert candidate_prev.shape == (hidden_size, 1)

    (
        w_forget,
        w_input,
        w_candidate,
        w_output,
        w_logit,
        b_forget,
        b_input,
        b_candidate,
        b_output,
        b_logit,
    ) = params

    x_s, z_s, f_s, i_s, = (
        [],
        [],
        [],
        [],
    )
    g_s, C_s, o_s, h_s = [], [], [], []
    v_s, output_s = [], []
    h_s.append(hidden_prev)
    C_s.append(candidate_prev)

    for x in inputs:
        z = np.row_stack((hidden_prev, x))
        z_s.append(z)

        forget = sigmoid(np.dot(w_forget, z) + b_forget)
        f_s.append(forget)

        input = sigmoid(np.dot(w_input, z) + b_input)
        i_s.append(input)

        g = tanh(np.dot(w_candidate, z) + b_candidate)
        g_s.append(g)

        candidate_prev = forget * candidate_prev + input * g
        C_s.append(candidate_prev)

        o = sigmoid(np.dot(w_output, z) + b_output)
        o_s.append(o)

        hidden_prev = o * tanh(candidate_prev)
        h_s.append(hidden_prev)

        v = np.dot(w_logit, hidden_prev) + b_logit
        v_s.append(v)

        output = softmax(v)
        output_s.append(output)

    return z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, output_s


def backward_pass_lstm(
    z_s: List[np.ndarray],
    f_s: List[np.ndarray],
    i_s: List[np.ndarray],
    g_s: List[np.ndarray],
    C_s: List[np.ndarray],
    o_s: List[np.ndarray],
    h_s: List[np.ndarray],
    outputs: List[np.ndarray],
    targets: List[np.ndarray],
    params: np.ndarray,
) -> Tuple[
    float,
    Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
]:
    """Backward pass function for LSTM network.

    Args:
        z_s (List[np.ndarray]): List of prev hidden state and input row stack outputs.
        f_s (List[np.ndarray]): List of forget gate outputs.
        i_s (List[np.ndarray]): List of input gate outputs.
        g_s (List[np.ndarray]): List of candidate outputs.
        C_s (List[np.ndarray]): List of memory cell outputs.
        o_s (List[np.ndarray]): List of output gate outputs.
        h_s (List[np.ndarray]): List of hidden state outputs.
        v_s (List[np.ndarray]): List of logit outputs.
        output_s (List[np.ndarray]): List of softmaxed logit outputs.
        targets (List[np.ndarray]): List of targets.
        params (np.ndarray): State of the network.
        hidden_size (int): Size of LSTM hidden state.

    Returns:
        loss (float): Loss for samples.
        grads (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]): Tuple of gradient matrices.
    """
    w_forget, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = params

    d_w_forget = np.zeros_like(w_forget)
    d_b_forget = np.zeros_like(b_f)
    d_w_input = np.zeros_like(W_i)
    d_b_input = np.zeros_like(b_i)
    d_w_candidate = np.zeros_like(W_g)
    d_b_candidate = np.zeros_like(b_g)
    d_w_output = np.zeros_like(W_o)
    d_b_output = np.zeros_like(b_o)
    d_w_logit = np.zeros_like(W_v)
    d_b_logit = np.zeros_like(b_v)

    d_hidden_next = np.zeros_like(h_s[0])
    d_memory_cell_next = np.zeros_like(C_s[0])

    loss = 0

    for t in reversed(range(len(outputs))):

        loss += -np.mean(np.log(outputs[t]) * targets[t])

        C_prev = C_s[t - 1]

        d_logit = np.copy(outputs[t])
        d_logit[np.argmax(targets[t])] -= 1

        d_w_logit += np.dot(d_logit, h_s[t].T)
        d_b_logit += d_logit

        d_hidden = np.dot(W_v.T, d_logit)
        d_hidden += d_hidden_next
        d_output = d_hidden * tanh(C_s[t])
        d_output = sigmoid(o_s[t], derivative=True) * d_output

        d_w_output += np.dot(d_output, z_s[t].T)
        d_b_output += d_output

        d_memory_cell = np.copy(d_memory_cell_next)
        d_memory_cell += d_hidden * o_s[t] * tanh(tanh(C_s[t]), derivative=True)
        d_candidate = d_memory_cell * i_s[t]
        d_candidate = tanh(g_s[t], derivative=True) * d_candidate

        d_w_candidate += np.dot(d_candidate, z_s[t].T)
        d_b_candidate += d_candidate

        d_input = d_memory_cell * g_s[t]
        d_input = sigmoid(i_s[t], True) * d_input
        d_w_input += np.dot(d_input, z_s[t].T)
        d_b_input += d_input

        d_forget = d_memory_cell * C_prev
        d_forget = sigmoid(f_s[t]) * d_forget
        d_w_forget += np.dot(d_forget, z_s[t].T)
        d_b_forget += d_forget

        dz = (
            np.dot(w_forget.T, d_forget)
            + np.dot(W_i.T, d_input)
            + np.dot(W_g.T, d_candidate)
            + np.dot(W_o.T, d_output)
        )

    grads = (
        d_w_forget,
        d_w_input,
        d_w_candidate,
        d_w_output,
        d_w_logit,
        d_b_forget,
        d_b_input,
        d_b_candidate,
        d_b_output,
        d_b_logit,
    )

    grads = clip_gradient_norm(grads)

    return loss, grads

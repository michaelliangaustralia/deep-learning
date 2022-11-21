import utils
import numpy as np

import IPython

# Network hyperparameters.
num_epochs = 1000

# Datasets.
sequences = utils.generate_dataset(num_sequences=100)
word_to_idx, idx_to_word, num_sequences, vocab_size = utils.sequences_to_dict(sequences)
training_set, validation_set, test_set = utils.create_datasets(sequences, utils.Dataset)

# Network initialization.
hidden_size = 50
z_size = hidden_size + vocab_size
params = utils.init_lstm(hidden_size, vocab_size, z_size)
training_loss, validation_loss = [], []


# Get first sentence in test set
inputs, targets = test_set[1]

# One-hot encode input and target sequence
inputs_one_hot = utils.one_hot_encode_sequence(inputs, vocab_size, word_to_idx)
targets_one_hot = utils.one_hot_encode_sequence(targets, vocab_size, word_to_idx)

# Initialize hidden state as zeros
h = np.zeros((hidden_size, 1))
c = np.zeros((hidden_size, 1))

# Forward pass
z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = utils.forward_pass_lstm(inputs_one_hot, h, c, params, hidden_size)

loss, grads = utils.backward_pass_lstm(z_s, f_s, i_s, g_s, C_s, o_s, h_s, outputs, targets_one_hot, params)
print(loss)

output_sentence = [idx_to_word[np.argmax(output)] for output in outputs]
print('Input sentence:')
print(inputs)

print('\nTarget sequence:')
print(targets)

print('\nPredicted sequence:')
print([idx_to_word[np.argmax(output)] for output in outputs])
import utils
import numpy as np

import IPython

# Network hyperparameters.
num_epochs = 50

# Datasets.
sequences = utils.generate_dataset(num_sequences=100)
word_to_idx, idx_to_word, num_sequences, vocab_size = utils.sequences_to_dict(sequences)
training_set, validation_set, test_set = utils.create_datasets(sequences, utils.Dataset)

# Network initialization.
hidden_size = 50
z_size = hidden_size + vocab_size
params = utils.init_lstm(hidden_size, vocab_size, z_size)
hidden_state = np.zeros((hidden_size, 1))

training_loss, validation_loss = [], []

for i in range(num_epochs):

    epoch_training_loss, epoch_validation_loss = 0, 0

    for inputs, targets in validation_set:
        inputs_one_hot = utils.one_hot_encode_sequence(inputs, vocab_size, word_to_idx)
        targets_one_hot = utils.one_hot_encode_sequence(
            targets, vocab_size, word_to_idx
        )

        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))

        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = utils.forward_pass_lstm(
            inputs_one_hot, h, c, params, hidden_size
        )

        loss, _ = utils.backward_pass_lstm(
            z_s, f_s, i_s, g_s, C_s, o_s, h_s, outputs, targets_one_hot, params
        )

        epoch_validation_loss += loss

    for inputs, targets in training_set:
        inputs_one_hot = utils.one_hot_encode_sequence(inputs, vocab_size, word_to_idx)
        targets_one_hot = utils.one_hot_encode_sequence(
            targets, vocab_size, word_to_idx
        )

        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))

        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = utils.forward_pass_lstm(
            inputs_one_hot, h, c, params, hidden_size
        )

        loss, grads = utils.backward_pass_lstm(
            z_s, f_s, i_s, g_s, C_s, o_s, h_s, outputs, targets_one_hot, params
        )

        params = utils.update_parameters(params, grads, lr=1e-1)

        epoch_training_loss += loss

    validation_loss.append(epoch_validation_loss / len(validation_set))
    training_loss.append(epoch_training_loss / len(training_set))

    if i % 5 == 0:
        print(
            f"Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}"
        )


# Get first sentence in test set
# Get first sentence in test set
inputs, targets = test_set[1]

# One-hot encode input and target sequence
inputs_one_hot = utils.one_hot_encode_sequence(inputs, vocab_size, word_to_idx)
targets_one_hot = utils.one_hot_encode_sequence(targets, vocab_size, word_to_idx)

# Initialize hidden state as zeros
h = np.zeros((hidden_size, 1))
c = np.zeros((hidden_size, 1))

# Forward pass
z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = utils.forward_pass_lstm(
    inputs_one_hot, h, c, params, hidden_size
)

# Print example
print("Input sentence:")
print(inputs)

print("\nTarget sequence:")
print(targets)

print("\nPredicted sequence:")
print([idx_to_word[np.argmax(output)] for output in outputs])

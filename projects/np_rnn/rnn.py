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
params = utils.init_rnn(hidden_size, vocab_size)
hidden_state = np.zeros((hidden_size, 1))

training_loss, validation_loss = [], []

# Training loop.
for i in range(num_epochs):

    epoch_training_loss = 0
    epoch_validation_loss = 0

    for inputs, targets in validation_set:

        inputs_one_hot = utils.one_hot_encode_sequence(inputs, vocab_size, word_to_idx)
        targets_one_hot = utils.one_hot_encode_sequence(
            targets, vocab_size, word_to_idx
        )

        hidden_state = np.zeros_like(hidden_state)

        outputs, hidden_states = utils.forward_pass(
            inputs_one_hot, hidden_state, params
        )

        loss, _ = utils.backward_pass(
            inputs_one_hot, outputs, hidden_states, targets_one_hot, params
        )

        epoch_validation_loss += loss

    for inputs, targets in training_set:

        inputs_one_hot = utils.one_hot_encode_sequence(inputs, vocab_size, word_to_idx)
        targets_one_hot = utils.one_hot_encode_sequence(
            targets, vocab_size, word_to_idx
        )

        hidden_state = np.zeros_like(hidden_state)

        outputs, hidden_states = utils.forward_pass(
            inputs_one_hot, hidden_state, params
        )

        loss, grads = utils.backward_pass(
            inputs_one_hot, outputs, hidden_states, targets_one_hot, params
        )

        if np.isnan(loss):
            raise ValueError("Gradients have vanished!")

        params = utils.update_parameters(params, grads, lr=3e-4)

        epoch_training_loss += loss

    training_loss.append(epoch_training_loss / len(training_set))
    validation_loss.append(epoch_validation_loss / len(validation_set))

    if i % 100 == 0:
        print(
            f"Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}"
        )

# Test
inputs, targets = test_set[1]

inputs_one_hot = utils.one_hot_encode_sequence(inputs, vocab_size, word_to_idx)
targets_one_hot = utils.one_hot_encode_sequence(targets, vocab_size, word_to_idx)

hidden_state = np.zeros((hidden_size, 1))

outputs, hidden_states = utils.forward_pass(inputs_one_hot, hidden_state, params)
output_sentence = [idx_to_word[np.argmax(output)] for output in outputs]
print("Input sentence:")
print(inputs)

print("\nTarget sequence:")
print(targets)

print("\nPredicted sequence:")
print([idx_to_word[np.argmax(output)] for output in outputs])

# Inference
print(
    utils.inference(
        params,
        vocab_size,
        hidden_size,
        idx_to_word,
        word_to_idx,
        sentence="a a a a a b",
    )
)

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
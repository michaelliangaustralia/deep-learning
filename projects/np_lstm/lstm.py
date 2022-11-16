import utils
import numpy as np

import IPython

sequences = utils.generate_dataset(n_sequences=100)

word_to_idx, idx_to_word, num_sequences, vocab_size = utils.sequences_to_dict(sequences)

training_set, validation_set, test_set = utils.create_datasets(sequences, utils.Dataset)

test_sentence = utils.one_hot_encode_sequence(['a', 'b'], vocab_size, word_to_idx)

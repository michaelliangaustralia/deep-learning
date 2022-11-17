import utils
import numpy as np

import IPython

sequences = utils.generate_dataset(n_sequences=100)

word_to_idx, idx_to_word, num_sequences, vocab_size = utils.sequences_to_dict(sequences)

training_set, validation_set, test_set = utils.create_datasets(sequences, utils.Dataset)

test_sentence = utils.one_hot_encode_sequence(['a', 'b'], vocab_size, word_to_idx)

hidden_size = 50

params = utils.init_rnn(hidden_size, vocab_size)

test_input_sequence, test_target_sequence = training_set[0]
test_input = utils.one_hot_encode_sequence(test_input_sequence, vocab_size, word_to_idx)
test_target = utils.one_hot_encode_sequence(test_target_sequence, vocab_size, word_to_idx)

hidden_state = np.zeros((hidden_size, 1))

outputs, hidden_states = utils.forward_pass(test_input, hidden_state, params)


IPython.embed()
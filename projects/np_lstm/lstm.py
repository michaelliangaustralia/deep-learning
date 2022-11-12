import utils
import numpy as np

import IPython

sequences = utils.generate_dataset(n_sequences = 100)

_ = utils.sequences_to_dict(sequences)
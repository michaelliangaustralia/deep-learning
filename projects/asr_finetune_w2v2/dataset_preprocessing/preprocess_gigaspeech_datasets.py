import torchaudio
import datasets
import torch
import json
import numpy as np
import re
from tqdm import tqdm

# Vocab
with open('../vocab_en.json') as f:
    vocab = json.load(f)

# Data
giga = datasets.load_dataset("speechcolab/gigaspeech", "l", use_auth_token=True)
giga = datasets.concatenate_datasets([giga['train'], giga['validation'], giga['test']])

# Remove columns
giga = giga.remove_columns(["segment_id", "speaker", "begin_time", "end_time", "audio_id", "title", "url", "source", "category", "original_full_path"])
giga = giga.rename_column("text", "labels")
giga = giga.rename_column("audio", "input_values")

# Preprocess
def preprocess(batch):
    # Audio
    audio = np.array(batch['input_values']['array'])
    audio = audio.astype('float32')


    # Preprocess label
    label = batch['labels']
    # remove punctuation
    label = label.replace("<COMMA>", "").replace("<PERIOD>", "").replace("<QUESTIONMARK>", "").replace("<EXCLAMATIONPOINT>", "")
    label = label.lower()

    new_label = ''
    for idx, c in enumerate(label):
        if c == " " or c in vocab:
            new_label += c

    # Output: input_values, labels, input_length
    new_batch = {}
    new_batch['input_values'] = list(audio)
    new_batch['labels'] = new_label
    new_batch['input_length'] = len(new_batch['input_values'])
    if new_batch['input_values'] == []:
        new_batch['input_values'] = [0]
    return new_batch


giga = giga.map(preprocess, num_proc=16)

giga.save_to_disk('../data/giga_train')


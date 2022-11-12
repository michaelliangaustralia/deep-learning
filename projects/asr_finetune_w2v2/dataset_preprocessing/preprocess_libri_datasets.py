import torchaudio
import datasets
import torch
import json
import numpy as np
# Vocab
with open('../vocab_en.json') as f:
    vocab = json.load(f)

# Data
libri = datasets.load_dataset("librispeech_asr", "clean")
libri = datasets.concatenate_datasets([libri['train.360'], libri['train.100']])
# libri = libri['validation']
# Remove columns
libri = libri.remove_columns(["speaker_id", "chapter_id", "id", "file"])
libri = libri.rename_column("text", "labels")
libri = libri.rename_column("audio", "input_values")

# Preprocess
def preprocess(batch):

    audio = np.array(batch['input_values']['array'])
    audio = audio.astype('float32')
    label = batch['labels']
    # Preprocess label
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

    return new_batch


libri = libri.map(preprocess, num_proc=32)

libri.save_to_disk('data/libri_train')


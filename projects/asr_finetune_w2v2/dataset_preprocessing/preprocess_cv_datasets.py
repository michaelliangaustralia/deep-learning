import torchaudio
import datasets
import torch
import json
import numpy as np

# Vocab
with open("../vocab_en.json") as f:
    vocab = json.load(f)

# Data
cv = datasets.load_dataset("common_voice", "en")
cv_train = cv["train"]

# Remove columns
cv_train = cv_train.remove_columns(
    [
        "client_id",
        "up_votes",
        "down_votes",
        "age",
        "gender",
        "accent",
        "locale",
        "segment",
        "path",
    ]
)

cv_train = cv_train.rename_column("sentence", "labels")
cv_train = cv_train.rename_column("audio", "input_values")

# Preprocess
resample = torchaudio.transforms.Resample(48000, 16000)


def preprocess(batch):

    audio = torch.from_numpy(batch["input_values"]["array"])
    label = batch["labels"]

    # Resample
    audio = resample(audio).squeeze().numpy()

    # Preprocess label
    label = label.lower()
    new_label = ""
    for idx, c in enumerate(label):
        if c == " " or c in vocab:
            new_label += c

    # Output: input_values, labels, input_length
    new_batch = {}
    new_batch["input_values"] = audio
    new_batch["labels"] = new_label
    new_batch["input_length"] = len(new_batch["input_values"])

    return new_batch


cv_train = cv_train.map(preprocess, num_proc=32)

cv_train.save_to_disk("data/cv_train")

''' 
Working from https://www.youtube.com/watch?v=MMkeLjcBTcI&list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm&index=9&t=217s

Ways to improve
- Time/freq masking

'''
from comet_ml import Experiment
with open('../comet_api_key.txt') as f:
    comet_api_key = f.readline()
    experiment = Experiment(
        comet_api_key,
        project_name="michaelliang-dev"
    )
import torch
import torchaudio
import model
import datasets
import pandas as pd
import datasets
from tqdm import tqdm
import itertools
import json


import IPython

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Data
BSIZE = 16

train_ds = datasets.load_dataset("csv", data_files="data/train_curated.csv", split="train")
train_ds = train_ds.rename_column("fname", "audio")

# Preprocess data
path_to_data = 'data/train_curated/'
def preprocess_data(row):
    # update path
    row['audio'] = path_to_data + row['audio']
    # make comma separate labels into list
    row['labels'] = row['labels'].split(',')
    return row

train_ds = train_ds.map(preprocess_data, num_proc=4)

# Classification dictionary
with open('labels.json') as f:
    labels_dict = json.load(f)

# Data loader and collator
def collate_fn(batch):
    # convert audio into mel spec
    for x in batch:
        audio = x['audio']
        wav, sr = torchaudio.load(audio, normalize=True)
        # resample
        if sr != 16000:
            resampler=torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=16000,
                dtype=wav.dtype
            )
            wav = resampler(wav)

        if wav.shape[-1] > 16000 * 5: # if wav is longer than 5 seconds then we trim it to 5 seconds, we make all audio data 5 seconds
            wav = wav[0,:16000*5]

        x['audio'] = wav
    audio_padded = torch.zeros(BSIZE, 16000 * 5, dtype=torch.float)
    labels_padded = torch.zeros(BSIZE, 80, dtype=torch.float)

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=80,
        win_length=50,
        hop_length=25
    )

    for idx, x in enumerate(batch):
        audio_padded[idx, :x['audio'].shape[-1]] = x['audio']
        mel_specgram = transform(audio_padded)
        x['labels'] = sorted([labels_dict[x] for x in x['labels']])
        for l in x['labels']:
            labels_padded[idx, l] = 1


    return (mel_specgram, labels_padded)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=BSIZE,
    num_workers=16,
    collate_fn=collate_fn,
    shuffle=True,
    # drop_last=True # I think incomplete last batch = exploding gradient problem
)

# Model
model = model.AudioTaggingModel().to(device)
epochs = 1000

# Loss criterion
criterion = torch.nn.BCELoss()

# Optimizer and LR scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-5, epochs=epochs, steps_per_epoch=len(train_loader))

# Train

for e in range(epochs):
    print(f'Training epoch {e+1}...')
    # log lr
    experiment.log_metric("lr", lr_scheduler.get_last_lr(), epoch=e+1)

    # Train loop
    for idx, batch in enumerate(tqdm(train_loader)):
        audio, labels = batch
        audio = audio.to(device)
        labels = labels.to(device)
    
        # clear stored gradients
        optimizer.zero_grad()

        # model
        outputs = model(audio)

        # loss
        loss = criterion(outputs, labels)
        # print('loss', loss) # don't print loss like a maniac

        # backpropgate
        loss.backward()

        # clip grad norm to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # this also helps with exploding gradients

        optimizer.step()

    lr_scheduler.step()

    # save model every 10th epoch
    if (e+1) % 10 == 0:
        torch.save(model.state_dict(), f'outputs/audio_classification_v3.pt')



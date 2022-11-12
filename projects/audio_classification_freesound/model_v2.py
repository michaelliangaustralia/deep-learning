'''
CNN encoder
linear decoder

Questions
-  what is the intuition for the conv2d numbers below?
- what is max pooling doing?
- what's the intuition behind increasing the # of channels
'''

from torch import nn
import torch

import IPython

class AudioTaggingModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, # audio can be thought of as a single channel image
                out_channels=16, 
                kernel_size=3, # I chose these numbers so that the dimensions would stay the same
                stride=1,
                padding=1
            ), # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2 # this halves the time dimension and the mel filterbank dimension
            )
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32, 
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2
            )
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64, 
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2
            )
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128, 
                kernel_size=3, 
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2
            )
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128000, 8096)
        self.linear2 = nn.Linear(8096, 1024)
        self.linear3 = nn.Linear(1024, 80)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)



    def forward(self, audio):
        audio = audio.unsqueeze(1) # N, C, H, W - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        x = self.conv1(audio)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.relu(self.linear1(x)) # the relus make all the difference in helping the network learn non-linearities?
        x = self.relu(self.linear2(x))
        x = self.linear3(x) # no relu on the last layer since we're doing classification
        x = self.softmax(x)
        return x
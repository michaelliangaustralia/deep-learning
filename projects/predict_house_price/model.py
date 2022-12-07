from torch import nn

import IPython


class HousePricePredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(335, 670)
        self.linear2 = nn.Linear(670, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, 670)
        self.linear5 = nn.Linear(670, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.relu(self.linear3(x))
        x = self.dropout(x)
        x = self.relu(self.linear4(x))
        x = self.linear5(x)

        return x

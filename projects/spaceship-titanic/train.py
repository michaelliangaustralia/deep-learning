"""Questions

- How to balance what should be done within the network vs what is done in dataset preprocessing?
- Try to do it with hf trainer.
"""
import datasets
import torch
import model
from tqdm import tqdm

import IPython

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset.
train_ds = datasets.load_dataset('csv', data_files="train.csv")['train']
test_ds = datasets.load_dataset('csv', data_files="test.csv")["train"]

# Preprocess dataset.
train_ds = train_ds.remove_columns(['PassengerId', 'Name'])

# Naive preprocessing of nulls by removing rows with nulls.
train_ds = train_ds.filter(lambda x: None not in x.values())
test_ds = test_ds.filter(lambda x: None not in x.values())

# Split cabin column into multiple data columns.
def split_cabin_column(row):
    cabin_data = row['Cabin'].split('/')
    row['Cabin_deck'] = cabin_data[0]
    row['Cabin_num'] = int(cabin_data[1])
    row['Cabin_side'] = cabin_data[2]
    return row

train_ds = train_ds.map(split_cabin_column)
test_ds = test_ds.map(split_cabin_column)

train_ds = train_ds.remove_columns(['Cabin'])

# Categorical encode categorical variables. TODO: Would one hot encoding work better?
total_ds = datasets.concatenate_datasets([train_ds, test_ds])
str_to_idx_HomePlanet = {v: i for i, v in enumerate(set(total_ds['HomePlanet']))}
str_to_idx_CryoSleep = {v: i for i, v in enumerate(set(total_ds['CryoSleep']))}
str_to_idx_Cabin_deck = {v: i for i, v in enumerate(set(total_ds['Cabin_deck']))}
str_to_idx_Cabin_side = {v: i for i, v in enumerate(set(total_ds['Cabin_side']))}
str_to_idx_Destination = {v: i for i, v in enumerate(set(total_ds['Destination']))}
str_to_idx_VIP = {v: i for i, v in enumerate(set(total_ds['VIP']))}
str_to_idx_Transported = {v: i for i, v in enumerate(set(total_ds['Transported']))}

def categorical_encode(row):
    row['HomePlanet'] = str_to_idx_HomePlanet[row['HomePlanet']]
    row['CryoSleep_int'] = str_to_idx_CryoSleep[row['CryoSleep']]
    row['Cabin_deck'] = str_to_idx_Cabin_deck[row['Cabin_deck']]
    row['Cabin_side'] = str_to_idx_Cabin_side[row['Cabin_side']]
    row['Destination'] = str_to_idx_Destination[row['Destination']]
    row['VIP_int'] = str_to_idx_VIP[row['VIP']]
    row['Transported_int'] = str_to_idx_Transported[row['Transported']]
    return row

train_ds = train_ds.map(categorical_encode)
train_ds = train_ds.remove_columns(['CryoSleep', 'VIP'])
train_ds = train_ds.rename_column('CryoSleep_int', 'CryoSleep')
train_ds = train_ds.rename_column('VIP_int', 'VIP')
train_ds = train_ds.rename_column('Transported_int', 'target')

# Convert all numerical variables into integers.
def convert_to_int(row):
    row['Age_int'] = int(row['Age'])
    row['RoomService_int'] = int(row['RoomService'])
    row['FoodCourt_int'] = int(row['FoodCourt'])
    row['ShoppingMall_int'] = int(row['ShoppingMall'])
    row['Spa_int'] = int(row['Spa'])
    row['VRDeck_int'] = int(row['VRDeck'])
    return row

train_ds = train_ds.map(convert_to_int)
train_ds = train_ds.remove_columns(['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])
train_ds = train_ds.rename_column('Age_int', 'Age')
train_ds = train_ds.rename_column('RoomService_int', 'RoomService')
train_ds = train_ds.rename_column('FoodCourt_int', 'FoodCourt')
train_ds = train_ds.rename_column('ShoppingMall_int', 'ShoppingMall')
train_ds = train_ds.rename_column('Spa_int', 'Spa')
train_ds = train_ds.rename_column('VRDeck_int', 'VRDeck')

# Dataloader.
def collate_fn(batch):
    source = torch.zeros(len(batch), 13)
    target = torch.zeros(len(batch), 1)

    for i, row in enumerate(batch):
        # Source.
        source[i, 0] = row['Age']
        source[i, 1] = row['HomePlanet']
        source[i, 2] = row['CryoSleep']
        source[i, 3] = row['Cabin_deck']
        source[i, 4] = row['Cabin_num']
        source[i, 5] = row['Cabin_side']
        source[i, 6] = row['Destination']
        source[i, 7] = row['VIP']
        source[i, 8] = row['RoomService']
        source[i, 9] = row['FoodCourt']
        source[i, 10] = row['ShoppingMall']
        source[i, 11] = row['Spa']
        source[i, 12] = row['VRDeck']
        # Target.
        target[i, 0] = row['target']
    return source, target

dataloader = torch.utils.data.DataLoader(train_ds, batch_size=32, collate_fn=collate_fn)

# Model.
n_epochs = 100
model = model.TitanicModel().to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(n_epochs):
    for batch in tqdm(dataloader):
        source, target = batch
        source = source.to(device)
        target = target.to(device)

        # Forward pass.
        output = model(source)

        # Backward pass.
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(loss)



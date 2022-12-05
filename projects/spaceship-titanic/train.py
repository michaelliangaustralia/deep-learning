"""Questions

- How to balance what should be done within the network vs what is done in dataset preprocessing?
- Try to do it with hf trainer.
- Would one hot encoding work better in encoding categorical variables?
"""
import datasets
import torch
import model
import utils
from tqdm import tqdm
import os
import subprocess

import IPython

# Create output folder if doesn't exist
if not os.path.exists('outputs'):
    os.makedirs('outputs')
else:
    subprocess.run("find outputs -mindepth 1 -delete", shell=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset.
ds = datasets.load_dataset('csv', data_files="train.csv")['train']
# Split train and validation.
ds = ds.train_test_split(test_size=0.1, seed=42)
ds['val'] = ds['test']
del ds['test']

# Fill all nones.
ds = ds.map(utils.fill_none)
ds = ds.remove_columns(['PassengerId', 'HomePlanet', 'Cabin', 'Destination', 'CryoSleep', 'VIP', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name'])

# Remove no null from all column names
ds = ds.rename_column('PassengerId_no_null', 'PassengerId')
ds = ds.rename_column('HomePlanet_no_null', 'HomePlanet')
ds = ds.rename_column('Cabin_no_null', 'Cabin')
ds = ds.rename_column('Destination_no_null', 'Destination')
ds = ds.rename_column('CryoSleep_no_null', 'CryoSleep')
ds = ds.rename_column('VIP_no_null', 'VIP')
ds = ds.rename_column('Age_no_null', 'Age')
ds = ds.rename_column('RoomService_no_null', 'RoomService')
ds = ds.rename_column('FoodCourt_no_null', 'FoodCourt')
ds = ds.rename_column('ShoppingMall_no_null', 'ShoppingMall')
ds = ds.rename_column('Spa_no_null', 'Spa')
ds = ds.rename_column('VRDeck_no_null', 'VRDeck')
ds = ds.rename_column('Name_no_null', 'Name')


# Split cabin column into multiple data columns.
ds = ds.map(utils.split_cabin_column, num_proc=1)
ds = ds.remove_columns(['Cabin'])

# Categorical encode categorical variables.
total_ds = datasets.concatenate_datasets([ds['train'], ds['val']])
str_to_idx_HomePlanet = utils.str_to_idx(total_ds, 'HomePlanet')
str_to_idx_CryoSleep = utils.str_to_idx(total_ds,'CryoSleep')
str_to_idx_Cabin_deck = utils.str_to_idx(total_ds,'Cabin_deck')
str_to_idx_Cabin_side = utils.str_to_idx(total_ds,'Cabin_side')
str_to_idx_Destination = utils.str_to_idx(total_ds,'Destination')
str_to_idx_VIP = utils.str_to_idx(total_ds,'VIP')
str_to_idx_Transported = utils.str_to_idx(total_ds,'Transported')

ds = ds.map(utils.categorical_encode, fn_kwargs = {
    'str_to_idx_dict': {
        'HomePlanet': str_to_idx_HomePlanet,
        'CryoSleep': str_to_idx_CryoSleep,
        'Cabin_deck': str_to_idx_Cabin_deck,
        'Cabin_side': str_to_idx_Cabin_side,
        'Destination': str_to_idx_Destination,
        'VIP': str_to_idx_VIP,
        'Transported': str_to_idx_Transported,
    }
    }, num_proc=1)

ds = ds.remove_columns(['CryoSleep', 'VIP'])
ds = ds.rename_column('CryoSleep_int', 'CryoSleep')
ds = ds.rename_column('VIP_int', 'VIP')
ds = ds.rename_column('Transported_int', 'target')

# Convert all numerical variables into integers.
ds = ds.map(utils.convert_to_int)
ds = ds.remove_columns(['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])
ds = ds.rename_column('Age_int', 'Age')
ds = ds.rename_column('RoomService_int', 'RoomService')
ds = ds.rename_column('FoodCourt_int', 'FoodCourt')
ds = ds.rename_column('ShoppingMall_int', 'ShoppingMall')
ds = ds.rename_column('Spa_int', 'Spa')
ds = ds.rename_column('VRDeck_int', 'VRDeck')


# Dataloader.
train_dataloader = torch.utils.data.DataLoader(ds['train'], batch_size=32, collate_fn=utils.collate_fn)
val_dataloader = torch.utils.data.DataLoader(ds['val'], batch_size=16, collate_fn=utils.collate_fn)

# Model.
n_epochs = 1000
model = model.TitanicModel().to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in tqdm(range(n_epochs)):
    train_loss = []
    # Train loop.
    for batch in train_dataloader:
        source, target = batch
        source = source.to(device)
        target = target.to(device)

        # Forward pass.
        output = model(source)

        # Backward pass.
        loss = criterion(output, target)
        train_loss.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loop.
    with torch.no_grad():
        val_loss = []
        for batch in val_dataloader:

            source, target = batch
            source = source.to(device)
            target = target.to(device)

            # Forward pass.
            output = model(source)

            # Loss calculation.
            loss = criterion(output, target)
            val_loss.append(loss)
    
    print(f'Epoch: {epoch}, Train Loss: {torch.mean(torch.stack(train_loss))}, Validation Loss: {torch.mean(torch.stack(val_loss))}')
    # Save model checkpoint.
    torch.save(model.state_dict(), f'outputs/model_{epoch}.pth')



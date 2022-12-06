import torch
import datasets
import utils
import model
import numpy as np
import pandas as pd
from tqdm import tqdm

import IPython

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model from checkpoint.
state_dict = torch.load('model_999.pth')
model = model.TitanicModel().to(device)
model.load_state_dict(state_dict)

# Load test dataset.
ds = datasets.load_dataset('csv', data_files="test.csv")['train']

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
ds = ds.map(utils.split_cabin_column)
ds = ds.remove_columns(['Cabin'])

# Categorical encode categorical variables.
str_to_idx_HomePlanet = utils.str_to_idx(ds, 'HomePlanet')
str_to_idx_CryoSleep = utils.str_to_idx(ds,'CryoSleep')
str_to_idx_Cabin_deck = utils.str_to_idx(ds,'Cabin_deck')
str_to_idx_Cabin_side = utils.str_to_idx(ds,'Cabin_side')
str_to_idx_Destination = utils.str_to_idx(ds,'Destination')
str_to_idx_VIP = utils.str_to_idx(ds,'VIP')

ds = ds.map(utils.categorical_encode, fn_kwargs = {
    'str_to_idx_dict': {
        'HomePlanet': str_to_idx_HomePlanet,
        'CryoSleep': str_to_idx_CryoSleep,
        'Cabin_deck': str_to_idx_Cabin_deck,
        'Cabin_side': str_to_idx_Cabin_side,
        'Destination': str_to_idx_Destination,
        'VIP': str_to_idx_VIP,
    }
    }, num_proc=1)
ds = ds.remove_columns(['CryoSleep', 'VIP'])
ds = ds.rename_column('CryoSleep_int', 'CryoSleep')
ds = ds.rename_column('VIP_int', 'VIP')

# Convert all numerical variables into integers.
ds = ds.map(utils.convert_to_int)
ds = ds.remove_columns(['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])
ds = ds.rename_column('Age_int', 'Age')
ds = ds.rename_column('RoomService_int', 'RoomService')
ds = ds.rename_column('FoodCourt_int', 'FoodCourt')
ds = ds.rename_column('ShoppingMall_int', 'ShoppingMall')
ds = ds.rename_column('Spa_int', 'Spa')
ds = ds.rename_column('VRDeck_int', 'VRDeck')

# Output submission file.
df = pd.DataFrame(columns=['PassengerId', 'Transported'])

# Inference.
for d in tqdm(ds):
    with torch.no_grad():
        source = utils.create_inference_input(d)
        source = source.to(device)

        # Forward pass.
        output = model(source)
        output = torch.round(output)

        new_row = {
            'PassengerId': d['PassengerId'],
            'Transported': bool(output.item())
        }

        # Add row to pandas dataframe
        df = df.append(new_row, ignore_index=True)

# Save pandas df to csv
df.to_csv('submission.csv', index=False)




import torch
import datasets
import utils

import IPython

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model from checkpoint.
model = torch.load('model_999.pth')

# Load test dataset.
ds = datasets.load_dataset('csv', data_files="test.csv")['train']

# Replace None's with NaN's.
ds = ds.map(utils.replace_none_with_nan)
IPython.embed()

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
    })
    
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

IPython.embed()
import torch

import IPython
def split_cabin_column(row):
    if row['Cabin'] is not None:
        cabin_data = row['Cabin'].split('/')
        row['Cabin_deck'] = cabin_data[0]
        row['Cabin_num'] = int(cabin_data[1])
        row['Cabin_side'] = cabin_data[2]
    return row

def str_to_idx(ds, column_name):
    return {v: i for i, v in enumerate(set(ds[column_name]))}    

def categorical_encode(row, str_to_idx_dict):
    for x in str_to_idx_dict:
        row[x] = str_to_idx_dict[x][row[x]]
        if x == 'VIP':
            row['VIP_int'] = str_to_idx_dict[x][row[x]]
        elif x == 'Transported':
            row['Transported_int'] = str_to_idx_dict[x][row[x]]
        elif x == 'CryoSleep':
            row['CryoSleep_int'] = str_to_idx_dict[x][row[x]]
    return row

def convert_to_int(row):
    row['Age_int'] = int(row['Age'])
    row['RoomService_int'] = int(row['RoomService'])
    row['FoodCourt_int'] = int(row['FoodCourt'])
    row['ShoppingMall_int'] = int(row['ShoppingMall'])
    row['Spa_int'] = int(row['Spa'])
    row['VRDeck_int'] = int(row['VRDeck'])
    return row

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

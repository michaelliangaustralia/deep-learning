import torch

import IPython
def split_cabin_column(row):
    row['Cabin_deck'] = 'nan'
    row['Cabin_num'] = 0
    row['Cabin_side'] = 'nan'
    if row['Cabin'] != 'nan':
        cabin_data = row['Cabin'].split('/')
        row['Cabin_deck'] = str(cabin_data[0])
        row['Cabin_num'] = int(cabin_data[1])
        row['Cabin_side'] = str(cabin_data[2])
    return row

def str_to_idx(ds, column_name):
    return {v: i for i, v in enumerate(set(ds[column_name]))}    

def categorical_encode(row, str_to_idx_dict):
    for x in str_to_idx_dict:
        if x == 'VIP':
            row['VIP_int'] = str_to_idx_dict[x][row[x]]
        elif x == 'Transported':
            row['Transported_int'] = str_to_idx_dict[x][row[x]]
        elif x == 'CryoSleep':
            row['CryoSleep_int'] = str_to_idx_dict[x][row[x]]
        else:
            row[x] = str_to_idx_dict[x][row[x]]
    return row

def convert_to_int(row):
    row['Age_int'] = int(float(row['Age'])) if row['Age'] != 'nan' else 0
    row['RoomService_int'] = int(float(row['RoomService'])) if row['RoomService'] != 'nan' else 0
    row['FoodCourt_int'] = int(float(row['FoodCourt'])) if row['FoodCourt'] != 'nan' else 0
    row['ShoppingMall_int'] = int(float(row['ShoppingMall'])) if row['ShoppingMall'] != 'nan' else 0
    row['Spa_int'] = int(float(row['Spa'])) if row['Spa'] != 'nan' else 0
    row['VRDeck_int'] = int(float(row['VRDeck'])) if row['VRDeck'] != 'nan' else 0
    return row

def collate_fn(batch):
    source = torch.zeros(len(batch), 13)
    target = torch.zeros(len(batch), 1)

    for i, row in enumerate(batch):
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

        target[i, 0] = row['target']
    return source, target

def fill_none(row):
    row['PassengerId_no_null'] = str(row['PassengerId']) if type(row['PassengerId']) != type(None) else "nan"
    row['HomePlanet_no_null'] = str(row['HomePlanet']) if type(row['HomePlanet']) != type(None) else "nan"
    row['CryoSleep_no_null'] = str(row['CryoSleep']) if type(row['CryoSleep']) != type(None) else "nan"
    row['Cabin_no_null'] = str(row['Cabin']) if type(row['Cabin']) != type(None) else "nan"
    row['Destination_no_null'] = str(row['Destination']) if type(row['Destination']) != type(None) else "nan"
    row['Age_no_null'] = str(row['Age']) if type(row['Age']) != type(None) else "nan"
    row['VIP_no_null'] = str(row['VIP'])  if type(row['VIP']) != type(None) else "nan"
    row['RoomService_no_null'] = str(row['RoomService']) if type(row['RoomService']) != type(None) else "nan"
    row['FoodCourt_no_null'] = str(row['FoodCourt']) if type(row['FoodCourt']) != type(None) else "nan"
    row['ShoppingMall_no_null'] = str(row['ShoppingMall']) if type(row['ShoppingMall']) != type(None) else "nan"
    row['Spa_no_null'] = str(row['Spa']) if type(row['Spa']) != type(None) else "nan"
    row['VRDeck_no_null'] = str(row['VRDeck']) if type(row['VRDeck']) != type(None) else "nan"
    row['Name_no_null'] = str(row['Name']) if type(row['Name']) != type(None) else "nan"
    return row
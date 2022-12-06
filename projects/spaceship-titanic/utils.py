import torch
import datasets
from typing import List, Dict, Tuple, Any

import IPython
def split_cabin_column(row: Dict[str, Any]) -> Dict[str, Any]:
    """Split the Cabin column on a delimiter /.

    Args:
        row (Dict[str, Any]): A row from the dataset

    Returns:
        row (Dict[str, Any]): The dataset row with the Cabin column split into Cabin_deck,
            Cabin_num and Cabin_side
    """
    row['Cabin_deck'] = 'nan'
    row['Cabin_num'] = 0
    row['Cabin_side'] = 'nan'
    if row['Cabin'] != 'nan':
        cabin_data = row['Cabin'].split('/')
        row['Cabin_deck'] = str(cabin_data[0])
        row['Cabin_num'] = int(cabin_data[1])
        row['Cabin_side'] = str(cabin_data[2])
    return row

def str_to_idx(ds: datasets.Dataset, column_name: str) -> datasets.Dataset:
    """Returns a dictionary mapping the unique values in a column to an index.
    
    Args:
        ds (datasets.Dataset): The dataset to convert
        column_name (str): The name of the column to convert
        
    Returns:
        row (datasets.Dataset): The dataset with the column converted to an index column
    """
    return {v: i for i, v in enumerate(set(ds[column_name]))}    

def categorical_encode(row: Dict[str, Any], str_to_idx_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Encodes categorical variables into integer mapping.
    
    Args:
        row (Dict[str, Any]): A row from the dataset
        str_to_idx_dict (Dict[str, Dict[str, Any]]): A dictionary mapping the unique values in a column to an index
        
    Returns:
        row (Dict[str, Any]): The dataset row with the categorical columns encoded
    """
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

def convert_to_int(row: Dict[str, Any]) -> Dict[str, Any]:
    """Converts numerical data to integers.

    Args:
        row (Dict[str, Any]): A row from the dataset

    Returns:
        row (Dict[str, Any]): The dataset row with the numerical columns converted to integers
    """
    row['Age_int'] = int(float(row['Age'])) if row['Age'] != 'nan' else 0
    row['RoomService_int'] = int(float(row['RoomService'])) if row['RoomService'] != 'nan' else 0
    row['FoodCourt_int'] = int(float(row['FoodCourt'])) if row['FoodCourt'] != 'nan' else 0
    row['ShoppingMall_int'] = int(float(row['ShoppingMall'])) if row['ShoppingMall'] != 'nan' else 0
    row['Spa_int'] = int(float(row['Spa'])) if row['Spa'] != 'nan' else 0
    row['VRDeck_int'] = int(float(row['VRDeck'])) if row['VRDeck'] != 'nan' else 0
    return row

def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collate function for the dataloader.
    
    Args:
        batch (List[Dict[str, Any]]): A list of dictionaries containing the data for a batch
        
    Returns:
        row st[Dict[str, Any]]): A list of dictionaries containing the source data for a batch
        target (List[Dict[str, Any]]): A list of dictionaries containing the target data for a batch
    """
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

def fill_none(row: Dict[str, Any]) -> Dict[str, Any]:
    """Fills None values with 'nan' and convert to string format.
    
    Args:
        row (Dict[str, Any]): A row from the dataset
    
    Returns:
        row (Dict[str, Any]): The dataset row with None values filled with 'nan'
    """
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

def create_inference_input(row: Dict[str, Any]) -> Dict[str, Any]:
    """Creates the input for the inference function.
    
    Args:
        row (Dict[str, Any]): A row from the dataset
        
    Returns:
        source (Dict[str, Any]): The dataset row with the columns renamed and the target column removed
    """
    source = torch.zeros(13)
    source[0] = row['Age']
    source[1] = row['HomePlanet']
    source[2] = row['CryoSleep']
    source[3] = row['Cabin_deck']
    source[4] = row['Cabin_num']
    source[5] = row['Cabin_side']
    source[6] = row['Destination']
    source[7] = row['VIP']
    source[8] = row['RoomService']
    source[9] = row['FoodCourt']
    source[10] = row['ShoppingMall']
    source[11] = row['Spa']
    source[12] = row['VRDeck']
    return source
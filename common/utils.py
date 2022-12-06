# Third party imports.
import torch
import IPython
import datasets

# Native imports.
import os
import subprocess
from typing import List, Dict, Tuple, Any, Optional

hf_dataset_row = datasets.arrow_dataset.Example

def get_device():
    """Returns the device to be used for training."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_outputs_folder():
    """Creates the outputs folder if it does not exist."""
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

def delete_outputs_folder_contents():
    """Deletes the contents of the outputs folder."""
    subprocess.run("find outputs -mindepth 1 -delete", shell=True)

def cast_hf_dataset_columns(ds: datasets.Dataset, feature: datasets.Features, column_names: Optional[List[str]] = None) -> datasets.Dataset:
    """Casts the columns of a HuggingFace dataset to the correct types.
    
    Args:
        ds (datasets.Dataset): The dataset to cast.
        feature (datasets.Features): The feature to cast the columns to.
        column_names (Optional[List[str]]): The list of column names to cast. Leave as None to cast all columns.
    
    Returns:
        ds (datasets.Dataset): The dataset with new features.
    """
    new_features = {}
    for f in ds.features.items():
        if column_names is None or f[0] in column_names:
            new_features[f[0]] = feature
        else:
            new_features[f[0]] = f[1]
    ds = ds.cast(datasets.Features(new_features))
    return ds

def get_dict_map(ds: datasets.Dataset, column_name: str) -> Dict[str, Any]:
    """Returns a dictionary mapping the unique values in a column to an index.
    
    Args:
        ds (datasets.Dataset): The dataset to convert
        column_name (str): The name of the column to convert
        
    Returns:
        row Dict[str, Any]: The dataset with the column converted to an index column
    """
    return {v: i for i, v in enumerate(set(ds[column_name]))}

def get_multiple_dict_maps(ds: datasets.Dataset, column_name_list: List[str]) -> Dict[str, Dict[str, Any]]:
    """Returns a dictionary mapping the unique values in a column to an index.
    
    Args:
        ds (datasets.Dataset): The dataset to convert
        column_name_list (List[str]): The list of names of the columns to convert
        
    Returns:
        dict_maps [Dict[Dict[str, Any]]]: The list of dictionary maps for each column
    """
    dict_maps = {}
    for column_name in column_name_list:
        dict_map = get_dict_map(ds, column_name)
        dict_maps[column_name] = dict_map
    return dict_maps

def one_hot_encode(row: hf_dataset_row, dict_maps: Dict[str, Dict[str, Any]], column_names: List[str]) -> hf_dataset_row:
    """One hot encodes a column in a dataset.
    
    Args:
        row (hf_dataset_row): The row of a dataset to one hot encode.
        dict_map (Dict[str, Dict[str, Any]]): The dictionary mapping the unique values in a column to an index.
        column_name (List[str]): The name of the column to one hot encode.
        
    Returns:
        row (hf_dataset_row): The row with new columns added.
    """
    for column_name in column_names:
        row_value = row[column_name]
        row[f'one_hot_{column_name}_{row_value}'] = 1
        for k in dict_maps[column_name]:
            if k != row_value:
                row[f'one_hot_{column_name}_{k}'] = 0
    return row
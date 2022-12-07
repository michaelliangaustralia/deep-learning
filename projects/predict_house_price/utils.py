import torch
import datasets
from typing import List, Dict, Tuple, Any
import IPython

def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collate function for the dataloader.
    
    Args:
        batch (List[Dict[str, Any]]): A list of dictionaries containing the data for a batch
        
    Returns:
        source [Dict[str, Any]]): A list of dictionaries containing the source data for a batch
        target (List[Dict[str, Any]]): A list of dictionaries containing the target data for a batch
    """
    source = torch.zeros(len(batch), 335)
    target = torch.zeros(len(batch), 1)
    for batch_idx, row in enumerate(batch):
        for idx, k in enumerate(row):
            source[batch_idx, idx] = int(row[k])
        target[batch_idx, 0] = row['SalePrice']
    return source, target
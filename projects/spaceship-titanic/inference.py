import torch
import datasets

# Load model from checkpoint.
model = torch.load('outputs/model_100.pth')

# Load test dataset.
test_ds = torch.load('outputs/test_ds.pth')

# Native imports.
import sys

sys.path.append("../..")  # Configuring sys path to enable imports from parent folder.
import common.utils as common_utils

# Third party imports
import torch
import datasets
import utils
import model
import pandas as pd
from tqdm import tqdm
import IPython

# Device
device = common_utils.get_device()

# Load model from checkpoint.
state_dict = torch.load("model.pth")
model = model.HousePricePredictionModel().to(device)
model.load_state_dict(state_dict)
model.eval()

# Load test dataset.
ds = datasets.load_from_disk("test_processed")

# Dataframe with int column type and float column type.
df = pd.DataFrame(columns=["Id", "SalePrice"])

# Inference.
test_dataloader = torch.utils.data.DataLoader(
    ds, batch_size=1, collate_fn=utils.collate_fn_inf
)

for batch in tqdm(test_dataloader):
    with torch.no_grad():
        source, id = batch
        source = source.to(device)

        # Forward pass.
        output = model(source)
        new_row = {"Id": int(id), "SalePrice": output.cpu().item()}

        # Add row to pandas dataframe
        df = df.append(new_row, ignore_index=True)

# Convert pandas df to correct types
df = df.astype({"Id": "int", "SalePrice": "float"})

# Save pandas df to csv
df.to_csv("submission.csv", index=False)

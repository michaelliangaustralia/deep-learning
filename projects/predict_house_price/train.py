# Native imports.
import sys
sys.path.append('../..') # Configuring sys path to enable imports from parent folder.
import common.utils as common_utils

# Third party imports
import datasets
import torch
import model
import utils
from tqdm import tqdm
import IPython

# Comet ML logging
common_utils.start_comet_ml_logging('michaelliang-dev')

# Create clean output folder
common_utils.create_outputs_folder()
common_utils.delete_outputs_folder_contents()

device = common_utils.get_device()

# Dataset.
ds = datasets.load_from_disk('train_processed')
ds = ds.train_test_split(test_size=0.1, seed=42)
ds['val'] = ds['test']
del ds['test']

# Dataloader.
train_dataloader = torch.utils.data.DataLoader(ds['train'], batch_size=16, collate_fn=utils.collate_fn)
val_dataloader = torch.utils.data.DataLoader(ds['val'], batch_size=16, collate_fn=utils.collate_fn)

# Model.
n_epochs = 1000
model = model.HousePricePredictionModel().to(device)
criterion = torch.nn.L1Loss()
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



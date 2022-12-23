# Local imports.
import model

# Third-party import
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from typing import Callable, Optional
import IPython
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import subprocess
import shutil

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make outputs folders
os.makedirs("outputs", exist_ok=True)
subprocess.run("find outputs -mindepth 1 -delete", shell=True)

# Make submission directory
os.makedirs("outputs/submission", exist_ok=True)
subprocess.run("find outputs/submission -mindepth 1 -delete", shell=True)


# Training and Model Hyperparameters
lr = 3e-4
batch_size = 16
img_dim = 3 * 256 * 256
num_epochs = 50

# Init the models
gen_pickle = model.Generator().to(device)
gen_raw = model.Generator().to(device)
disc_pickle = model.Discriminator().to(device)
disc_raw = model.Discriminator().to(device)


# Datasets
class ImageDataset(Dataset):
    """Image dataset.
    
    Loads images from a directory.
    """
    def __init__(self, train_dir: str, test_dir: str, transforms: Optional[Callable] = None):
        """Initializes the ImageDataset.
        
        Args:
            train_dir (str): Directory containing the raw training images.
            test_dir (str): Directory containing the raw training images.
            transforms (Optional[Callable]): Optional transform to be applied on a sample.
        """
        self.train_dir = sorted(glob.glob(os.path.join(train_dir, "*.jpg")))
        self.test_dir = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
        self.transforms = transforms

    def __getitem__(self, index):
        train_image = Image.open(self.train_dir[index % len(self.train_dir)])
        test_image = Image.open(self.test_dir[index % len(self.test_dir)])
        if self.transforms:
            train_image = self.transforms(train_image)
            test_image = self.transforms(test_image)
        return train_image, test_image

    def __len__(self):
        return len(self.train_dir)
# Image Transforms
transforms = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])
# Create the dataset
ds = ImageDataset(train_dir='data/photo_jpg/', test_dir='data/pickle_jpg/', transforms=transforms)

# Create the data loader
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)        


# Optimizers
opt_gen = torch.optim.Adam(list(gen_pickle.parameters()) + list(gen_raw.parameters()), lr=lr)
opt_disc = torch.optim.Adam(list(disc_pickle.parameters()) + list(disc_raw.parameters()), lr=lr)

# Loss
L1 = nn.L1Loss() # Cycle consistency loss and identity loss
mse = nn.MSELoss() # Adversarial loss

step = 0
# Training Loop
for epoch in range(num_epochs):
    for idx, (raw_img, pickle_img) in enumerate(tqdm(dataloader)):
        raw_img = raw_img.to(device)
        pickle_img = pickle_img.to(device)
        
        # Train discriminators pickle and Raw
        fake_pickle = gen_pickle(raw_img)
        D_pickle_real = disc_pickle(pickle_img)
        D_pickle_fake = disc_pickle(fake_pickle.detach())
        D_pickle_real_loss = mse(D_pickle_real, torch.ones_like(D_pickle_real))
        D_pickle_fake_loss = mse(D_pickle_fake, torch.zeros_like(D_pickle_fake))
        D_pickle_loss = D_pickle_real_loss + D_pickle_fake_loss

        fake_raw = gen_raw(pickle_img)
        D_raw_real = disc_raw(raw_img)
        D_raw_fake = disc_raw(fake_raw.detach())
        D_raw_real_loss = mse(D_raw_real, torch.ones_like(D_raw_real))
        D_raw_fake_loss = mse(D_raw_fake, torch.zeros_like(D_raw_fake))
        D_raw_loss = D_raw_real_loss + D_raw_fake_loss

        D_loss = (D_pickle_loss + D_raw_loss)/2

        opt_disc.zero_grad()
        D_loss.backward(retain_graph=True)
        opt_disc.step()

        # Train generators pickle and Raw
        D_pickle_fake = disc_pickle(fake_pickle)
        D_raw_fake = disc_raw(fake_raw)
        loss_G_pickle = mse(D_pickle_fake, torch.ones_like(D_pickle_fake)) # Adversarial loss
        loss_G_raw = mse(D_raw_fake, torch.ones_like(D_raw_fake)) # Adversarial loss

        # Cycle loss
        cycle_pickle = gen_pickle(fake_raw)
        cycle_raw = gen_raw(fake_pickle)
        cycle_pickle_loss = L1(pickle_img, cycle_pickle)
        cycle_raw_loss = L1(raw_img, cycle_raw)

        G_loss = (loss_G_pickle + loss_G_raw) + (cycle_pickle_loss + cycle_raw_loss)
        
        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        step += 1
        if idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {idx}/{len(dataloader)} \
                  Loss D: {D_loss:.4f}, loss G: {G_loss:.4f}")
            
            with torch.no_grad():
                fake = gen_pickle(raw_img)
                # Unnormalise the image
                raw_img = raw_img.reshape(-1, 3, 256, 256)
                raw_img = raw_img * 0.5 + 0.5
                fake = fake.reshape(-1, 3, 256, 256)
                fake = fake * 0.5 + 0.5
                # Concatenate raw and fake images
                img_grid = torch.cat((raw_img, fake), dim=0)
                # Save image
                save_image(img_grid, f"outputs/pickle_{step}.png")
       

    # Can we somehow see what the images look like as we train on tensorboard?
    # Save model
    torch.save(gen_pickle.state_dict(), f'outputs/gen_pickle_{epoch}.pth')
    torch.save(gen_raw.state_dict(), f'outputs/gen_raw_{epoch}.pth')
    torch.save(disc_pickle.state_dict(), f'outputs/disc_pickle_{epoch}.pth')
    torch.save(disc_raw.state_dict(), f'outputs/disc_raw_{epoch}.pth')

# Build submission
gen_pickle.eval()

for file in tqdm(sorted(glob.glob(os.path.join("data/photo_jpg/", "*.jpg")))):
    file_name = file.split("/")[-1]
    img = Image.open(file)
    img = transforms(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    pred = gen_pickle(img)
    pred = pred.reshape(-1, 3, 256, 256)
    pred = pred * 0.5 + 0.5
    save_image(pred, f"outputs/submission/{file_name}")

# Zip submission folder
shutil.make_archive("images", "zip", "outputs/submission")


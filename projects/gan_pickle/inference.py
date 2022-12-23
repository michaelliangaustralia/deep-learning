# Native imports.
import sys
sys.path.append("../..")
import common.utils as common_utils

# Local imports.
import model

# Third party imports.
import torch
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as transforms
import IPython
from tqdm import tqdm

# Device
device = common_utils.get_device()

# Load model
gen_pickle = model.Generator().to(device)
gen_pickle.eval()

# Image transforms
transforms = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])

# Load image
image = transforms(Image.open("IMG_1164.jpg")).to(device)

for i in tqdm(range(50)):
    gen_pickle.load_state_dict(torch.load(f"checkpoints/gen_monet_{i}.pth"))

    # Generate fake pickle
    fake_pickle = gen_pickle(image)

    # Save image.
    save_image(fake_pickle, f"test/fake_pickle_{i}.jpg")
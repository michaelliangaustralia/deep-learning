# Native imports.
import sys
sys.path.append("../..")
import common.utils as common_utils
import model

# Third party imports.
import torch
from torchvision.utils import save_image

# Device
device = common_utils.get_device()

# Load model state dict
z_dim = 64
image_dim = 28 * 28 * 1
gen = model.Generator(z_dim, image_dim).to(device)
gen.load_state_dict(torch.load("gen_49.pth", map_location=device))

# Generate fake images.
noise = torch.randn(1, z_dim).to(device)
fake = gen(noise)

# Save images.
fake = fake.view(28, 28)
save_image(fake, "fake.png")
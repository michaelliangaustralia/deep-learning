# Native imports
import sys

sys.path.append("../..")
import common.utils as common_utils
import model

# Third-party imports
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
import IPython
from tqdm import tqdm

# Make output directory
common_utils.delete_outputs_folder_contents()
common_utils.create_outputs_folder()

device = common_utils.get_device()
lr = 3e-4
batch_size = 32
z_dim = 64
image_dim = 28 * 28 * 1
num_epochs = 50

disc = model.Discriminator(image_dim).to(device)
gen = model.Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = torch.optim.Adam(disc.parameters(), lr=lr)
opt_gen = torch.optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

# Tensorboard.
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

step = 0

for epoch in tqdm(range(num_epochs)):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Discriminator.
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)

        # Real discriminator loss.
        disc_real = disc(real).view(
            -1
        )  # values should as close to 1 as possible since they are real.
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        # Fake discriminator loss.
        disc_fake = disc(fake).view(
            -1
        )  # values should as close to 0 as possible since they are fake.
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # Combine the loss.
        lossD = (lossD_real + lossD_fake) / 2

        # Backward pass.
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        # Generator.
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))

        # Backward pass.
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        # Tensorboard.
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                        Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1

        # Save the model.
        torch.save(gen.state_dict(), f"outputs/gen_{epoch}.pth")
        torch.save(disc.state_dict(), f"outputs/disc_{epoch}.pth")

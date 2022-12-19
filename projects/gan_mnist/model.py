import torch.nn as nn

class Discriminator(nn.Module):
    """Discriminator network.
    
    Attempts to classify real and fake images from the dataset and the Generator network.
    """
    def __init__(self, img_dim: int):
        """Initializes the Discriminator network.

        Args:
            img_dim (int): Dimension of the input space.
        """
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    """Generator network.
    
    Attempts to generate fake images that look like the real images from the dataset.
    """
    def __init__(self, z_dim: int, img_dim: int):
        """Initializes the Generator network.
        
        Args:
            z_dim (int): Dimension of the latent space.
            img_dim (int): Dimension of the image space.
        """
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim), 
            nn.Tanh() # For images, we want the values to be between -1 and 1.
        )

    def forward(self, x):
        return self.gen(x)
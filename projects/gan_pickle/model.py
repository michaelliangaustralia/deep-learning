# Third-party imports
import torch.nn as nn
import torch


class Block(nn.Module):
    """Basic building block of the Generator and Discriminator networks.
    
    Consists of a linear layer, batch normalization and LeakyReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """Initializes the Block.
        
        Args:
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
            stride (int): Stride of conv block.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    """Discriminator network.
    """
    def __init__(self, in_channels: int = 3, features: int = [64, 128, 256, 512]):
        """Initializes the Discriminator network.
        
        Args:
            in_channels (int): Number of channels in the input image.
            features (list): Number of features in each layer.
        """
        super().__init__()
        # Initial does not use instance norm in cycleGAN paper.
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))

class ConvBlock(nn.Module):
    """Building block for generator.
    """
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        """Initializes the ConvBlock.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            down (bool): Whether to use a downsample or upsample.
            use_act (bool): Whether to use an activation function.
            **kwargs: Additional arguments for the convolutional layer.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) if down else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )
        self.down = down

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    """Residual block for generator.
    """
    def __init__(self, channels):
        """Initializes the ResidualBlock.
        
        Args:
            channels (int): Number of channels in the input and output.
        """
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """Generator network.
    
    Attempts to augoment an input image to look like a pickle painting.
    """
    def __init__(self, img_channels: int = 3, num_features: int = 64, num_residuals: int = 9):
        """Initializes the Generator network.
        
        Args:
            img_channels (int): Dimension of the image space.
            num_features (int): Number of features to use.
            num_residuals (int): Number of residual blocks to use.
        """
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList([
            ConvBlock(num_features, num_features*2, down=True, kernel_size=3, stride=2, padding=1),
            ConvBlock(num_features*2, num_features*4, down=True, kernel_size=3, stride=2, padding=1),
        ])
        self.residual_blocks = nn.Sequential(*[ResidualBlock(num_features*4) for _ in range(num_residuals)])
        self.up_blocks = nn.ModuleList([
            ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(num_features*2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
        ])
        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.initial(x)
        for down_block in self.down_blocks:
            x = down_block(x)
        x = self.residual_blocks(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        return torch.tanh(self.last(x))
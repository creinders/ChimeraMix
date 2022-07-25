import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/models.py


class Discriminator(nn.Module):
    def __init__(self, input_shape, ndf=64):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, ndf, normalize=False),
            *discriminator_block(ndf, 2 * ndf),
            *discriminator_block(2 * ndf, 4 * ndf),
            *discriminator_block(4 * ndf, 8 * ndf),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(8 * ndf, 1, 4, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img, targets):
        return self.model(img)

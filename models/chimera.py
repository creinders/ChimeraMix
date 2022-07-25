import torch.nn as nn
import torch.nn.functional as F
import torch

# from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/models.py


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features=None,
        norm=nn.InstanceNorm2d,
        bias=True,
        kernel=3,
        pad=1,
    ):
        super(ResidualBlock, self).__init__()

        if out_features is None:
            out_features = in_features

        layers = [
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_features, out_features, kernel, bias=bias),
        ]

        if norm:
            layers += [norm(out_features)]

        layers += [
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(pad),
            nn.Conv2d(out_features, out_features, kernel, bias=bias),
        ]

        if norm:
            layers += [norm(out_features)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f_x, f_y, mask):

        mask_a = mask[:, 0, ...][:, None]
        mask_b = mask[:, 1, ...][:, None]

        f = f_x * mask_a + f_y * mask_b
        return f


class ChimeraEncoder(nn.Module):
    def __init__(self, num_residual_blocks, ngf, nc=3):
        super().__init__()

        norm = nn.InstanceNorm2d
        use_bias = norm == nn.InstanceNorm2d

        out_features = ngf
        model_downsampling = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nc, out_features, 7, bias=use_bias),
            norm(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model_downsampling += [
                nn.Conv2d(
                    in_features, out_features, 3, stride=2, padding=1, bias=use_bias
                ),
                norm(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        blocks = []

        # Residual blocks
        for i in range(num_residual_blocks):
            blocks += [ResidualBlock(out_features, norm=norm, bias=use_bias)]

        self.model_downsampling = nn.Sequential(*model_downsampling)
        self.model_blocks = nn.Sequential(*blocks)
        self.out_features = out_features

    def forward(self, x):
        f = self.model_downsampling(x)
        f = self.model_blocks(f)

        return f


class ChimeraDecoder(nn.Module):
    def __init__(self, num_residual_blocks, ngf):
        super().__init__()

        norm = nn.InstanceNorm2d
        use_bias = nn == nn.InstanceNorm2d

        output_channels = 3
        num_upsampling_blocks = 2
        in_features = ngf * 2 ** num_upsampling_blocks
        out_features = in_features

        blocks = []

        # Residual blocks
        for i in range(num_residual_blocks):
            blocks += [ResidualBlock(out_features, norm=norm, bias=use_bias)]

        model_upsampling = []
        # Upsampling
        for _ in range(num_upsampling_blocks):
            out_features //= 2
            model_upsampling += [
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=use_bias,
                ),
                norm(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model_upsampling += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_features, output_channels, 7),
            nn.Tanh(),
        ]

        self.model_blocks = nn.Sequential(*blocks)
        self.model_upsampling = nn.Sequential(*model_upsampling)

    def forward(self, x):
        f = self.model_blocks(x)
        f = self.model_upsampling(x)
        return f


class ChimeraModel(nn.Module):
    def __init__(self, nc, num_residual_blocks, split_index, image_size, ngf=64):
        super().__init__()

        num_encoder_blocks = split_index
        num_decoder_blocks = num_residual_blocks - split_index
        self.encoder = ChimeraEncoder(
            num_residual_blocks=num_encoder_blocks, ngf=ngf, nc=nc
        )
        self.decoder = ChimeraDecoder(num_residual_blocks=num_decoder_blocks, ngf=ngf,)
        self.mixer = Mixer()

    def encode(self, x):
        return self.encoder(x)

    def mixDecode(self, f_x, f_y, noise):

        f_mix = self.mixer(f_x, f_y, noise)

        mix_hat = self.decoder(f_mix)
        x_hat = self.decoder(f_x)
        y_hat = self.decoder(f_y)

        ms = []
        a = torch.cat((noise, 1 - noise), dim=1)
        ms.append(a)

        return mix_hat, x_hat, y_hat, ms

    def forward(self, x, y, noise):

        f_x = self.encode(x)
        f_y = self.encode(y)

        return self.mixDecode(f_x, f_y, noise)

        return mix_hat, x_hat, y_hat, ms


if __name__ == "__main__":

    from torchinfo import summary

    size = 96  # 96, 32

    model = ChimeraModel(nc=3, num_residual_blocks=4, image_size=size, split_index=2)
    summary(
        model,
        [(1, 3, size, size), (1, 3, size, size), (1, 1, size // 4, size // 4)],
        depth=20,
    )


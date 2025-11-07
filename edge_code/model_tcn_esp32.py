import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
            groups=groups, bias=True
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlockLite(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU6(inplace=True)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU6(inplace=True)

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        res = self.residual(x) if self.residual is not None else x
        return out + res


class EncoderLite(nn.Module):
    def __init__(self, input_dim, hidden_dims: List[int], kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, out_ch in enumerate(hidden_dims):
            in_ch = input_dim if i == 0 else hidden_dims[i - 1]
            dilation = 2 ** i
            self.layers.append(TCNBlockLite(in_ch, out_ch, kernel_size, dilation))
        self.receptive_field = self._calc_rf(kernel_size, len(hidden_dims))

    def _calc_rf(self, k, levels):
        return 1 + 2 * (k - 1) * (2**levels - 1)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x


class DecoderLite(nn.Module):
    def __init__(self, latent_dim, hidden_dims: List[int], output_dim, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, out_ch in enumerate(hidden_dims):
            in_ch = latent_dim if i == 0 else hidden_dims[i - 1]
            dilation = 2 ** (len(hidden_dims) - i - 1)
            self.layers.append(TCNBlockLite(in_ch, out_ch, kernel_size, dilation))
        self.out_proj = nn.Conv1d(hidden_dims[-1], output_dim, 1)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return self.out_proj(x)


class TCNAutoencoderLite(nn.Module):
    def __init__(
        self,
        input_dim=16,
        latent_dim=8,
        encoder_hidden=[16, 24],
        decoder_hidden=[24],
        kernel_size=3
    ):
        super().__init__()
        self.encoder = EncoderLite(input_dim, encoder_hidden, kernel_size)
        self.to_latent = nn.Conv1d(encoder_hidden[-1], latent_dim, 1)
        self.from_latent = nn.Conv1d(latent_dim, decoder_hidden[0], 1)
        self.decoder = DecoderLite(decoder_hidden[0], decoder_hidden[1:] + [input_dim], input_dim, kernel_size)
        self.receptive_field = self.encoder.receptive_field

        print(f"Lite TCN Autoencoder:")
        print(f"  Receptive field: {self.receptive_field} frames")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def encode(self, x):
        x = self.encoder(x)
        return self.to_latent(x)

    def decode(self, latent):
        x = self.from_latent(latent)
        return self.decoder(x)

    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return latent, recon
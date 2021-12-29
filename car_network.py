import torch.nn as nn
import torch

class conv_block(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(conv_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(self.out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(self.out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        out = self.conv_1(x)
        out = self.conv_2(out)

        return out


class fc_conv_block(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(fc_conv_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        out = self.conv(x)

        return out


class output_block(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(output_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1 = nn.Sequential(
            nn.Conv1d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(64, self.out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(self.out_channels, affine=True),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.conv_1(x)
        out = self.conv_2(x)

        return out


class downsample_res_block(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(downsample_res_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels == 64:
            self.bottleneck1 = nn.Identity()
        else:
            self.bottleneck1 = nn.Sequential(
                nn.Conv1d(self.in_channels, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(64, affine=True),
                nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        self.bottleneck2 = nn.Sequential(
            nn.Conv1d(64, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm1d(self.out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x1 = self.bottleneck1(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.bottleneck2(x4)
        x6 = self.shortcut(x)
        out = x5 + x6

        return out


class deconv_block(nn.Module):

    def __init__(self, in_channels, out_channels, padding, outpadding):

        super(deconv_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.outpadding = outpadding

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(self.in_channels, self.out_channels, kernel_size=3, stride=2, padding=self.padding, output_padding=self.outpadding, bias=False),
            nn.BatchNorm1d(self.out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        out = self.deconv(x)

        return out


class RegistrationNet(nn.Module):

    def __init__(self, moving_channel, target_channel):

        super(RegistrationNet, self).__init__()

        self.moving_channel = moving_channel
        self.target_channel = target_channel

        self.moving_input = conv_block(self.moving_channel, 64)
        self.target_input = conv_block(self.target_channel, 64)

        self.fuse = conv_block(128, 64)

        self.conv128 = downsample_res_block(64, 128)
        self.conv256 = downsample_res_block(128, 256)
        self.conv512 = downsample_res_block(256, 512)

        self.deconv256 = deconv_block(512, 256, 1, 1)
        self.deconv128 = deconv_block(256, 128, 1, 0)
        self.deconv64 = deconv_block(128, 64, 1, 0)

        self.moving_fc_conv_8 = fc_conv_block(self.moving_channel, 8)
        self.moving_fc_conv_16 = fc_conv_block(8, 16)
        self.moving_fc_conv_32 = fc_conv_block(16, 32)
        self.moving_fc_conv_64 = fc_conv_block(32, 64)

        self.target_fc_conv_8 = fc_conv_block(self.target_channel, 8)
        self.target_fc_conv_16 = fc_conv_block(8, 16)
        self.target_fc_conv_32 = fc_conv_block(16, 32)
        self.target_fc_conv_64 = fc_conv_block(32, 64)

        self.output = output_block(64, 2)

    def forward(self, sample, start):

        moving_sample = sample[:, :3, :]
        target_sample = sample[:, 3:, :]
        moving_start = start[:, :, :3]
        target_start = start[:, :, 3:]

        moving_input = self.moving_input(moving_sample)
        target_input = self.target_input(target_sample)

        moving_extra_8 = self.moving_fc_conv_8(moving_start.permute(0, 2, 1))
        moving_extra_16 = self.moving_fc_conv_16(moving_extra_8)
        moving_extra_32 = self.moving_fc_conv_32(moving_extra_16)
        moving_extra_64 = self.moving_fc_conv_64(moving_extra_32)

        target_extra_8 = self.target_fc_conv_8(target_start.permute(0, 2, 1))
        target_extra_16 = self.target_fc_conv_16(target_extra_8)
        target_extra_32 = self.target_fc_conv_32(target_extra_16)
        target_extra_64 = self.target_fc_conv_64(target_extra_32)

        moving = moving_input + moving_extra_64.expand(-1, -1, moving_input.size(-1))
        target = target_input + target_extra_64.expand(-1, -1, target_input.size(-1))

        down_64 = self.fuse(torch.cat((moving, target), 1))
        down_128 = self.conv128(down_64)
        down_256 = self.conv256(down_128)
        down_512 = self.conv512(down_256)
        up_256 = self.deconv256(down_512) + down_256
        up_128 = self.deconv128(up_256) + down_128
        up_64 = self.deconv64(up_128) + down_64

        out = self.output(up_64)

        return out
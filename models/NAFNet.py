import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from argparse import Namespace
from models import register
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)




def channel_shuffle(x, groups=2):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 1, stride=1, padding=0),
            nn.Conv2d(1, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2))

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class NAFBlock(nn.Module):
    def __init__(self, c, FFN_Expand=1.5):
        super().__init__()
        ffn_channel = int(np.ceil(c*FFN_Expand))
        self.conv1 = nn.Sequential(
            LayerNorm2d(c//2),
            nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=True),
            nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=3, padding=1, stride=1, groups=c // 2,
                      bias=True),
            nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=True),
            nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=3, padding=1, stride=1, groups=c // 2,
                      bias=True),
        )
        self.conv2 = nn.Sequential(
            LayerNorm2d(c//2),
            nn.Conv2d(in_channels=c // 2, out_channels=ffn_channel // 2, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=True),
            nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c // 2, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=True),
            SpatialGate()
        )



    def forward(self, inp):
        xr,xl = torch.chunk(inp,2,1)
        xr = self.conv1(xr)+xr
        xr = self.conv2(xr)+xr
        out = channel_shuffle(torch.cat((xl,xr),dim=1))
        return out


class NAFNet(nn.Module):

    def __init__(self,args, img_channel=1, width=16, middle_blk_num=2, enc_blk_nums=[1, 1], dec_blk_nums=[1, 1]):
        super().__init__()
        self.out_dim = 4

        self.intro = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels=img_channel * 4, out_channels=width, kernel_size=1, padding=0, stride=1,
                      groups=1,
                      bias=True),
            nn.Conv2d(in_channels=width, out_channels=width, kernel_size=[3, 3], padding=[1, 1],
                      stride=1,
                      groups=width,
                      bias=True),
            nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1, padding=0, stride=1,
                      groups=1,
                      bias=True),
            nn.Conv2d(in_channels=width, out_channels=width, kernel_size=[3, 3], padding=[1, 1],
                      stride=1,
                      groups=width,
                      bias=True),

        )
        self.ending = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=8, kernel_size=1, padding=0, stride=1,
                      groups=1,
                      bias=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=[3, 3], padding=[1, 1],
                      stride=1,
                      groups=8,
                      bias=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, padding=0, stride=1,
                      groups=1,
                      bias=True),
            nn.Conv2d(in_channels=8, out_channels=4*4, kernel_size=[3, 3], padding=[1, 1],
                      stride=1,
                      groups=8,
                      bias=True),
            nn.PixelShuffle(2)
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(chan, 2 * chan, 1, 1, 0),
                    nn.Conv2d(2 * chan, 2 * chan, [3, 3], [2, 2], [1, 1], groups=chan),
                )
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** (len(self.encoders)+1)

    def forward(self, inp):
        B, C, H, W = inp.shape
        # print(H,W)
        inp = self.check_image_size(inp)
        # print(inp.shape)
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp
        # print(x.shape)

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # print(self.padder_size)
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x



class Linearmlp(nn.Module):
    def __init__(self, c, FFN_Expand=1.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.ReLU(),
            nn.Linear(32, 1),
        )



    def forward(self, inp):
        inp = inp.permute(0,2,3,1)
        out = self.conv(inp)
        out = out.permute(0,3,1,2)
        return out

@register('NAFNet')
def make_edsr(n_resblocks=32, n_feats=256, res_scale=0.1,
              scale=2, no_upsampling=False, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 1
    return NAFNet(args)


if __name__ == '__main__':
    args = None
    net = NAFNet(args)
    print(net)
    print("params:", sum(param.numel() for param in net.parameters()))

    inp = torch.randn((1, 1, 64, 64))
    a = time.time()

    out = net(inp)
    print(time.time() - a)

    print(out.shape)

    inp_shape = (1, 64, 64)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, as_strings=False, print_per_layer_stat=True,
                                             verbose=True)
    print('Macs:  ', macs / 64 / 64)
    print('Params: ', params)

    # print(macs, params)

import math

import torch.nn as nn

from models import register
from .NAFNet import *

@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

@register('sinmlp')
class SinMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        layers.append(SineLayer(lastv,hidden_list[0],is_first=True))
        for hidden in hidden_list[0:]:
            lastv = hidden
            layers.append(SineLayer(lastv, hidden))
            # layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_list[0], out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

@register('fouriermlp')
class fourierMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        in_dim=44
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.LeakyReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):


        shape = x.shape[:-1]
        # print(x.shape)
        x = 2 * torch.tensor(math.pi) * x
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        # print(x.shape)
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


if __name__ == '__main__':
    args = None
    net = SinMLP(in_dim=4,out_dim=1,hidden_list=[16,16])
    print(net)
    print("params:", sum(param.numel() for param in net.parameters()))

    inp = torch.randn((1,  64, 64,4))
    a = time.time()

    out = net(inp)
    print(time.time() - a)

    print(out.shape)

    inp_shape = (64, 64,4)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, as_strings=False, print_per_layer_stat=True,
                                             verbose=True)
    print('Macs:  ', macs / 64 / 64)
    print('Params: ', params)
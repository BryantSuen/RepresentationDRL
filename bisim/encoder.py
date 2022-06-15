from turtle import forward
import torch
import torch.nn as nn
from bisim.basic_models import *

class ResnetEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.model = ResNet18(num_classes=feature_dim)

    def forward(self, obs):
        out = self.model(obs)
        return out


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim=256, num_layers=2, num_filters=32, stride=None):
        '''
        obs_shape: torch[channel, h, w]
        '''
        super().__init__()

        assert(obs_shape[0] == 4)

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=obs_shape[0], out_channels=num_filters, kernel_size=3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        # TODO: output size should be modified. (done)
        out_dim = {2: 39, 4: 35, 6: 31}[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        # self.outputs = dict()

    # def reparameterize(self, mu, logstd):
    #     std = torch.exp(logstd)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

    # def forward_conv(self, obs):
    #     self.outputs['obs'] = obs

    #     conv = torch.relu(self.convs[0](obs))
    #     self.outputs['conv1'] = conv

    #     for i in range(1, self.num_layers):
    #         conv = torch.relu(self.convs[i](conv))
    #         self.outputs['conv%s' % (i + 1)] = conv

    #     h = conv.view(conv.size(0), -1)
    #     return h

    def forward(self, obs, detach=False):
        h = obs
        for idx in range(self.num_layers):
            h = torch.relu(self.convs[idx](h))
        h = h.view(h.size(0), -1)

        if detach:
            h = h.detach()
            
        h_fc = self.fc(h)
        # self.outputs['fc'] = h_fc

        out = self.ln(h_fc)
        # self.outputs['ln'] = out
        return out

if __name__ == "__main__":
    pass
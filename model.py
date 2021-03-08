import torch
import torchvision
import torch.nn.init
import torch.nn as nn
from loss import Get_gradient


class VGG(nn.Module):
    def __init__(self, module='features', layer='15', pretrained=False):
        super().__init__()
        self.module = module
        self.layer = layer
        self.features = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),   # V
                                      # conv1

                                      nn.Conv2d(64, 128, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      # conv2

                                      nn.Conv2d(128, 256, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(inplace=True),  # V
                                      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                                      # conv3

                                      nn.Conv2d(256, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      # conv4

                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),              # V
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                      # conv5
                                      )

        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 2),
        )

        self.init_conv2d()
        self.reward_hook()

        if pretrained:

            std = torchvision.models.vgg16(pretrained=True).features.state_dict()
            model_dict = self.features.state_dict()
            pretrained_dict = {k: v for k, v in std.items() if k in model_dict}  # 여기서 orderdict 가 아니기 때문에
            model_dict.update(pretrained_dict)
            self.features.load_state_dict(model_dict, strict=True)

        self.features_m = nn.ModuleList(list(self.features.children()))  # nn.ModuleList

    def init_conv2d(self):
        for c in self.features.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def reward_hook(self):

        for modue_name, module in self._modules.items():
            if modue_name == self.module:
                for layer_name, module in module._modules.items():
                    if layer_name == self.layer:
                        module.register_forward_hook(self.forward_hook)
                        module.register_backward_hook(self.backward_hook)

    def forward_hook(self, _, input, output):
        self.forward_result = torch.squeeze(output)

    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = torch.squeeze(grad_output[0])

    def forward(self, x):
        for i in range(int(self.layer) + 1):
            x = self.features_m[i](x)
        k = x
        for i in range(int(self.layer) + 1, len(self.features_m)):
            x = self.features_m[i](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, k


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


def default_conv(in_channels, out_channels, bias=True):
    return nn.Conv2d(in_channels, out_channels, 3, padding=2, bias=bias, dilation=2)


class Block(nn.Module):
    def __init__(self, conv, dim):
        super(Block, self).__init__()

        self.conv1 = conv(dim, dim, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):

        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x

        return res


class ResGroup(nn.Module):
    def __init__(self, conv, dim, blocks):
        super(ResGroup, self).__init__()

        modules = [Block(conv, dim) for _ in range(blocks)]
        modules.append(conv(dim, dim))

        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


class Dual_Grad_Desnow_Net(nn.Module):
    """Encoder-Decoder architecture for AtDeSnow."""
    def __init__(self):
        super().__init__()

        self.feature_num = 48
        conv = default_conv
        self.edge = Get_gradient()
        self.g1 = nn.Sequential(
            nn.Conv2d(5, self.feature_num, 3, 1, 1, bias=True),
            ResGroup(conv, self.feature_num, blocks=3),
            nn.Conv2d(self.feature_num, 1, 3, 1, 1, bias=True),
        )
        self.g2 = nn.Sequential(
            nn.Conv2d(4, self.feature_num, 3, 1, 1, bias=True),
            ResGroup(conv, self.feature_num, blocks=20)
        )
        self.g3 = ResGroup(conv, self.feature_num, blocks=20)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.feature_num * 2, self.feature_num // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_num // 16, self.feature_num * 2, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.palayer = PALayer(self.feature_num)
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.feature_num, self.feature_num, 3, 1, 1, bias=True),
            nn.Conv2d(self.feature_num, 3, 3, 1, 1, bias=True),
        )
        self.decrease_object_edge = nn.Sequential(
            nn.Conv2d(3, self.feature_num, 3, 1, 1, bias=True),
            ResGroup(conv, self.feature_num, blocks=3),
            nn.Conv2d(self.feature_num, 1, 3, 1, 1, bias=True),
        )

    def forward(self, x, grad_guide):

        edge_guide = self.edge(x)
        edge_guide = self.decrease_object_edge(edge_guide)
        mask_out = self.g1(torch.cat([edge_guide, grad_guide, x], 1))

        res2 = self.g2(torch.cat([x - mask_out, mask_out], 1))
        res3 = self.g3(res2)

        w = self.ca(torch.cat([res2, res3], dim=1))
        out = w[:, 0:self.feature_num, :, :] * res2 + w[:, self.feature_num:self.feature_num*2, :, :] * res3
        out = out.contiguous()

        out = self.palayer(out)
        out = self.out_conv(out) + (x - mask_out)
        return out, mask_out, edge_guide


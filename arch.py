from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models import *
import functools
import operator



class nvidia(nn.Module):

    def __init__(self, input_channel, input_dim):
        super(nvidia, self).__init__()

        self.conv = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(input_channel, 24, 5, 2)),
                    ('elu1', nn.ELU()),
                    ('conv2', nn.Conv2d(24, 36, 5, 2)),
                    ('elu2', nn.ELU()),
                    ('conv3', nn.Conv2d(36, 48, 5, 2)),
                    ('elu3', nn.ELU()),
                    ('dropout', nn.Dropout()),
                    ('conv4', nn.Conv2d(48, 64, 3, 1)),
                    ('elu4', nn.ELU()),
                    ('conv5', nn.Conv2d(64, 64, 3, 1)),
                    ('elu5', nn.ELU()),
                ]))

        num_features_before_fcnn = functools.reduce(operator.mul, list(self.conv(torch.rand(1, *(input_channel, input_dim[0], input_dim[1]))).shape))

        self.linear = nn.Sequential(OrderedDict([
                    ('affine1', nn.Linear(num_features_before_fcnn, 100)),
                    ('elu6', nn.ELU()),
                    ('affine2', nn.Linear(100, 50)),
                    ('elu7', nn.ELU()),
                    ('affine3', nn.Linear(50, 10)),
                    ('elu8', nn.ELU()),
                    ('affine4', nn.Linear(10, 1)),
                ]))

    def forward(self, x):

        out = self.conv(x)
        feature = functools.reduce(operator.mul, out.shape[1:], 1)
        out = self.linear(out.view(-1, feature))
        return out.squeeze(1)

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight.data, mode = 'fan_in')
            if layer.bias is not None:
                layer.bias.data.zero_()



class MyConv(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.conv = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(9, 4, 5, 2)),
                    ('relu1', nn.ReLU()),
                    ('conv2', nn.Conv2d(4, 2, 3, 2)),
                    ('relu2', nn.ReLU()),
                    #('conv3', nn.Conv2d(8, 4, 3, 2)),
                    #('relu3', nn.ReLU()),
                    #('conv4', nn.Conv2d(4, 2, 3, 2)),
                    #('relu4', nn.ReLU()),
                    ('BN1', nn.BatchNorm2d(2, affine=False)),
                ]))

        num_features_before_fcnn = functools.reduce(operator.mul, list(self.conv(torch.rand(1, *(9, input_dim, input_dim))).shape))

        self.linear = nn.Sequential(OrderedDict([
                ('flatten', nn.Flatten()),
                ('affine1', nn.Linear(num_features_before_fcnn, 1024)),
                ('relu1', nn.ReLU()),
                ('affine2', nn.Linear(1024, 256)),
                ('relu2', nn.ReLU()),
                ('BN1', nn.BatchNorm1d(256, affine = False)),
                #('affine3', nn.Linear(256, 64)),
                ('affine4', nn.Linear(256, 1)),
            ]))


    def forward(self, x):
        out = self.conv(x)
        out = self.linear(out)
        return out


class ConvRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = wide_resnet50_2(pretrained=True).eval()

        self.rnn = nn.Sequential(OrderedDict([
        ('rnn1', nn.RNN(input_size = 1000, hidden_size = 64, num_layers = 10, nonlinearity = 'relu', batch_first = True, dropout = 0.3, bidirectional = True)),
        ]))

        self.linear = nn.Sequential(OrderedDict([
        ('flatten', nn.Flatten()),
        ('affine1', nn.Linear(64 * 2, 32)),
        ('relu1', nn.ReLU()),
        ('BN1', nn.BatchNorm1d(32, affine = False)),
        ('affine2', nn.Linear(32, 1)),
        ]))



    def forward(self, x):
        with torch.no_grad():
            out = torch.stack([self.resnet50(x[:,0:3,:,:]), self.resnet50(x[:,3:6,:,:]), self.resnet50(x[:,6:9,:,:])], dim=1)
        out, h_n = self.rnn(out)
        out = self.linear(out[:,-1,])
        return out



class ResConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = wide_resnet50_2(pretrained=True).eval()

        self.linear = nn.Sequential(OrderedDict([
        #('flatten', nn.Flatten()),
        ('affine1', nn.Linear(3000, 1000)),
        ('relu1', nn.ReLU()),
        ('affine2', nn.Linear(1000, 256)),
        ('relu2', nn.ReLU()),
        ('affine3', nn.Linear(256, 64)),
        ('relu3', nn.ReLU()),
        ('BN1', nn.BatchNorm1d(64, affine = False)),
        ('affine4', nn.Linear(64, 1)),
        ]))


    def forward(self, x):
        with torch.no_grad():
            out = torch.cat([self.resnet50(x[:,0:3,:,:]), self.resnet50(x[:,3:6,:,:]), self.resnet50(x[:,6:9,:,:])], dim=1)
        #print(out.shape)
        out = self.linear(out)
        return out


class opticalFlowModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(3, 5, 5, 2)),
                    ('relu1', nn.ReLU()),
                    ('conv2', nn.Conv2d(5, 3, 5, 2)),
                    ('relu2', nn.ReLU()),
                    ('avgpool', nn.AvgPool2d(3, stride=2)),
                    #('conv3', nn.Conv2d(3, 5, 5, 2)),
                    #('relu3', nn.ReLU()),
                    #('conv4', nn.Conv2d(5, 2, 5, 2)),
                    #('relu4', nn.ReLU()),
                    ('BN1', nn.BatchNorm2d(3, affine=False)),
                    ('flatten', nn.Flatten()),
                ]))

        num_features_before_fcnn = functools.reduce(operator.mul, list(self.conv(torch.rand(1, *(3, 480, 480))).shape))

        self.linear = nn.Sequential(OrderedDict([
                    #('flatten', nn.Flatten()),
                    ('affine1', nn.Linear(2000 + num_features_before_fcnn, 1000)),
                    ('relu1', nn.ReLU()),
                    ('affine2', nn.Linear(1000, 128)),
                    ('relu2', nn.ReLU()),
                    #('affine3', nn.Linear(256, 64)),
                    #('relu3', nn.ReLU()),
                    ('BN1', nn.BatchNorm1d(128, affine = False)),
                    ('affine4', nn.Linear(128, 1)),
                ]))


    def forward(self, opt, x):

        out = self.conv(opt)
        #print(out.shape)
        #print(x.shape)
        out = self.linear(torch.cat([out, x], dim=1))


        return out



class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Sequential(OrderedDict([
        ('affine1', nn.Linear(3000, 1000)),
        ('relu1', nn.ReLU()),
        ('affine2', nn.Linear(1000, 256)),
        ('relu2', nn.ReLU()),
        ('affine3', nn.Linear(256, 64)),
        ('relu3', nn.ReLU()),
        ('BN1', nn.BatchNorm1d(64, affine = False)),
        ('affine4', nn.Linear(64, 1)),
        ]))

    def forward(self, x):
        out = self.linear(x)
        return out



class half_UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(half_UNet, self).__init__()

        features = init_features
        self.bott = nn.Sequential(
            half_UNet._block(in_channels, features, name="enc1"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            half_UNet._block(features, features * 2, name="enc2"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            half_UNet._block(features * 2, features * 4, name="enc3"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            half_UNet._block(features * 4, features * 8, name="enc4"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            half_UNet._block(features * 8, features * 16, name="bottleneck"),
            nn.Flatten(),
        )

        num_features_before_fcnn = functools.reduce(operator.mul, list(self.bott(torch.rand(1, *(3, 200, 66))).shape))

        self.linear = nn.Sequential(
            nn.Linear(num_features_before_fcnn, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):

        x = self.bott(x)
        feature = functools.reduce(operator.mul, x.shape[1:], 1)
        out = self.linear(x.view(-1, feature))
        return out.squeeze(1)


    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

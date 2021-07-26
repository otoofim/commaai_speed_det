from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models import *
import functools
import operator





class nvidia(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(3, 24, 5, 2)),
                    ('conv2', nn.Conv2d(24, 36, 5, 2)),
                    ('conv3', nn.Conv2d(36, 48, 3, 1)),
                    ('conv4', nn.Conv2d(48, 64, 3, 1)),
                    ('conv5', nn.Conv2d(64, 64, 3, 1)),
                ]))

    def forward(self, x):
        out = self.conv(x)
        print(out.shape)
        #return out

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
        self.encoder1 = half_UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = half_UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = half_UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = half_UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = half_UNet._block(features * 8, features * 16, name="bottleneck")
        self.flatten = nn.Flatten()
        self.affine1 = nn.Linear(614400, 512)
        self.relu_aff = nn.ReLU()
        self.affine2 = nn.Linear(512, 1)


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.flatten(self.bottleneck(self.pool4(enc4)))
        print(bottleneck.shape)
        affi1 = self.relu_aff(self.affine1(bottleneck))
        out = self.affine2(affi1)
        return out


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

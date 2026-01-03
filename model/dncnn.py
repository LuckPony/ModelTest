import torch
import torch.nn as nn
class DnCNN(nn.Module):
    def __init__(self, channels = 1, num_layers = 17, features = 64):
        super(DnCNN, self).__init__()
        layers = []

        #第一层：Conv2D + ReLU
        layers.append(nn.Conv2d(channels,features,kernel_size=5,padding=2,bias=True))
        layers.append(nn.ReLU(inplace=True))#inplace=True表示直接在原变量上操作，节省内存

        #中间层：Conv2D + BatchNorm3D + ReLU
        for _ in range(num_layers-2):
            layers.append(nn.Conv2d(features,features,kernel_size=5,padding=2,bias=False))#BatchNorm 本身包含可学习的缩放（γ）和平移（β）参数,因此不需要bias
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        #最后一层：Conv2D输出噪声
        layers.append(nn.Conv2d(features,channels,kernel_size=5,padding=2,bias=True))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        out = x - noise
        return out
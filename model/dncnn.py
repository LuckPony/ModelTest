import torch
import torch.nn as nn
class DnCNN(nn.Module):
    def __init__(self, channels = 1, num_layers = 17, features = 64):
        super(DnCNN, self).__init__()
        layers = []

        #第一层：Conv3D + ReLU
        layers.append(nn.Conv3d(channels,features,kernel_size=3,padding=1,bias=True))
        layers.append(nn.ReLU(inplace=True))#inplace=True表示直接在原变量上操作，节省内存

        #中间层：Conv3D + BatchNorm3D + ReLU
        for _ in range(num_layers-2):
            layers.append(nn.Conv3d(features,features,kernel_size=3,padding=1,bias=False))#BatchNorm 本身包含可学习的缩放（γ）和平移（β）参数,因此不需要bias
            layers.append(nn.BatchNorm3d(features))
            layers.append(nn.ReLU(inplace=True))
        
        #最后一层：Conv3D输出噪声
        layers.append(nn.Conv3d(features,channels,kernel_size=3,padding=1,bias=True))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        out = x - noise
        return out
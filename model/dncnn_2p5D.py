import torch
import torch.nn as nn

class DnCNN_2p5D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_layers=17, features=64):
        """
        in_channels: 输入的相邻切片数量，默认 3 (Z-1, Z, Z+1)
        out_channels: 输出的去噪切片数量，默认 1 (只输出中心层 Z 的去噪结果)
        """
        super(DnCNN_2p5D, self).__init__()
        layers = []

        # 第一层：输入通道数为 in_channels (例如3)
        layers.append(nn.Conv2d(in_channels, features, kernel_size=5, padding=2, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # 中间层：提取 2.5D 融合特征
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=5, padding=2, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        # 最后一层：输出通道数为 out_channels (例如1，只预测中心层的噪声)
        layers.append(nn.Conv2d(features, out_channels, kernel_size=5, padding=2, bias=True))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # x 的维度应为 [Batch, in_channels, H, W]
        # 例如: [16, 3, 128, 128]
        
        # 网络预测出中心层的噪声
        noise = self.dncnn(x) # noise 维度为 [Batch, 1, H, W]

        # 动态获取中心层的索引
        # 如果输入 3 层，center_idx 是 1；如果输入 5 层，center_idx 是 2
        center_idx = x.shape[1] // 2 
        
        # 从输入中把中心层单独切片取出来作为基准
        # 注意切片方式 x[:, center_idx:center_idx+1, :, :] 可以保持维度为 [Batch, 1, H, W]
        center_slice = x[:, center_idx:center_idx+1, :, :] #左闭右开

        # 残差学习：中心层带噪图像 - 预测的中心层噪声
        out = center_slice - noise
        
        return out
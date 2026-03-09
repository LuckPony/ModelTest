import torch
import torch.nn as nn

class DnCNN_Decoupled(nn.Module):
    def __init__(self, num_directions=288, num_layers=12, features=64):
        """
        num_directions: 数据的总梯度方向数 (这里固定为 288)
        num_layers: 网络的总深度
        features: 隐藏层提取的特征通道数
        """
        super(DnCNN_Decoupled, self).__init__()
        
        # 模块一：角域信息交互模块 (Angular Interaction Module)
        # 使用 1x1 卷积跨越所有 288 个梯度方向提取非线性物理特征
        self.angular_module = nn.Sequential(
            nn.Conv2d(num_directions, features, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # 模块二：空间上下文提取模块 (Spatial Context Module)
        # 接收角域融合后的特征，在 2D 物理空间上进行去噪
        spatial_layers = []
        for _ in range(num_layers - 2):
            spatial_layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            spatial_layers.append(nn.BatchNorm2d(features))
            spatial_layers.append(nn.ReLU(inplace=True))
            
        # 最后一层：将特征映射回原有的 288 个方向，输出所有方向的预测噪声
        spatial_layers.append(nn.Conv2d(features, num_directions, kernel_size=3, padding=1, bias=True))
        
        self.spatial_module = nn.Sequential(*spatial_layers)

    def forward(self, x):
        # 输入 x 维度: [Batch, 288, H, W]
        
        # 1. 跨梯度方向融合
        feat = self.angular_module(x) # 维度变为: [Batch, features, H, W]
        
        # 2. 空间特征去噪
        noise = self.spatial_module(feat) # 维度变回: [Batch, 288, H, W]
        
        # 3. 残差学习
        out = x - noise
        
        return out
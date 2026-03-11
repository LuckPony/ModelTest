import torch
import torch.nn as nn

class ShellAwareDnCNN(nn.Module):
    def __init__(self, num_b0=18, num_b1000=90, num_b2000=90, features=64, num_layers=12):
        super(ShellAwareDnCNN, self).__init__()
        self.num_b0 = num_b0
        self.num_b1000 = num_b1000
        self.num_b2000 = num_b2000
        
        # ==========================================
        # 第一阶段：壳层独立提取 (Shell-Specific Extraction)
        # 作用：消除 b 值差异带来的巨大数值鸿沟
        # ==========================================
        self.b0_extract = nn.Sequential(
            nn.Conv2d(num_b0, features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.b1000_extract = nn.Sequential(
            nn.Conv2d(num_b1000, features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.b2000_extract = nn.Sequential(
            nn.Conv2d(num_b2000, features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # ==========================================
        # 第二阶段：深层联合融合 (Deep Joint Fusion)
        # 作用：在 192 维 (64*3) 的纯净特征空间内进行稠密推理
        # ==========================================
        fusion_layers = []
        fusion_channels = features * 3  # 64 + 64 + 64 = 192 维特征
        
        for _ in range(num_layers - 2):
            fusion_layers.append(nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1, bias=False))
            fusion_layers.append(nn.BatchNorm2d(fusion_channels))
            fusion_layers.append(nn.ReLU(inplace=True))
            
        self.fusion_module = nn.Sequential(*fusion_layers)

        # ==========================================
        # 第三阶段：噪声联合重建 (Joint Noise Reconstruction)
        # 作用：将特征重新映射回 198 个物理方向的纯噪声
        # ==========================================
        total_directions = num_b0 + num_b1000 + num_b2000
        self.noise_reconstruct = nn.Conv2d(fusion_channels, total_directions, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # 假设输入 x 的通道已经按照 b=0, b=1000, b=2000 排序好
        # x shape: [Batch, 198, H, W]
        
        # 1. 物理壳层分离
        x_b0 = x[:, :self.num_b0, :, :]
        x_b1000 = x[:, self.num_b0 : self.num_b0+self.num_b1000, :, :]
        x_b2000 = x[:, self.num_b0+self.num_b1000 :, :, :]

        # 2. 独立特征提取
        feat_b0 = self.b0_extract(x_b0)
        feat_b1000 = self.b1000_extract(x_b1000)
        feat_b2000 = self.b2000_extract(x_b2000)

        # 3. 特征级拼接 (Cat): [Batch, 192, H, W]
        feat_fused = torch.cat([feat_b0, feat_b1000, feat_b2000], dim=1)

        # 4. 空间-角域联合去噪推理
        feat_processed = self.fusion_module(feat_fused)

        # 5. 预测 198 个方向的噪声并执行残差减法
        noise = self.noise_reconstruct(feat_processed)
        out = x - noise
        
        return out
    

#     #            x
#             │
#    ┌────────┼────────┐
#    │        │        │
#   b0      b1000    b2000
#    │        │        │
# extract  extract   extract
#    │        │        │
#    └──────concat─────┘
#            │
#         fusion
#            │
#      noise reconstruction
#            │
#          output
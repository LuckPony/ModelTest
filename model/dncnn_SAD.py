import torch
import torch.nn as nn

class AngularBlock(nn.Module):
    def __init__(self, n_b0=18, n_b1000=90, n_b2000=90):
        super().__init__()
        # 1x1 卷积：绝对不降维，保留方向身份，分壳层独立提取
        self.b0_mix = nn.Sequential(nn.Conv2d(n_b0, n_b0, 1, bias=False), nn.BatchNorm2d(n_b0), nn.ReLU(inplace=True), nn.Conv2d(n_b0, n_b0, 1, bias=False))
        self.b1k_mix = nn.Sequential(nn.Conv2d(n_b1000, n_b1000, 1, bias=False), nn.BatchNorm2d(n_b1000), nn.ReLU(inplace=True), nn.Conv2d(n_b1000, n_b1000, 1, bias=False))
        self.b2k_mix = nn.Sequential(nn.Conv2d(n_b2000, n_b2000, 1, bias=False), nn.BatchNorm2d(n_b2000), nn.ReLU(inplace=True), nn.Conv2d(n_b2000, n_b2000, 1, bias=False))

    def forward(self, x):
        # x shape: [Batch * Slice(3), 198, H, W]
        x_b0 = x[:, :18, :, :]
        x_b1k = x[:, 18:108, :, :]
        x_b2k = x[:, 108:, :, :]

        # 【核心】：必须有残差连接！保留原始物理方向的基底
        out_b0 = x_b0 + self.b0_mix(x_b0)
        out_b1k = x_b1k + self.b1k_mix(x_b1k)
        out_b2k = x_b2k + self.b2k_mix(x_b2k)

        return torch.cat([out_b0, out_b1k, out_b2k], dim=1)


class Spatial2p5DBlock(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        # 2.5D 空域块：只看局部空间，输入 3 层输出 3 层
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        # x shape: [Batch * Direction(198), 3, H, W]
        return x + self.net(x)

 
class AS_Alternating_2p5D(nn.Module):
    def __init__(self, num_blocks=1, hidden_spatial=16):    #暂时把num_blocks设置为1，先跑一遍融合；hidden_spatial暂时由32设置为16（改这里不管用，要改训练文件）
        super().__init__()
        self.blocks = nn.ModuleList()
        # 交替堆叠 A-S 模块
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'angular': AngularBlock(),
                'spatial': Spatial2p5DBlock(hidden=hidden_spatial)
            }))
        
        # 最后的 Head：将 3 层的特征融合，直接预测中心层的 198 维噪声
        self.head = nn.Conv2d(198 * 3, 198, kernel_size=3, padding=1)

    def forward(self, x):
        # 输入维度: [Batch, Slices(3), Directions(198), H, W]
        B, S, Dd, H, W = x.shape
        feat = x
        
        for block in self.blocks:
            # 1. 切入角域态：折叠 3层 到 Batch
            feat_A = feat.view(B * S, Dd, H, W)
            feat_A = block['angular'](feat_A)
            
            # 2. 切入空域态：恢复 -> 交换维度 -> 折叠 198方向 到 Batch
            feat_S = feat_A.view(B, S, Dd, H, W).transpose(1, 2).contiguous()
            feat_S = feat_S.view(B * Dd, S, H, W)
            feat_S = block['spatial'](feat_S)
            
            # 3. 恢复为交替态：恢复 -> 交换维度 -> [B, 3, 198, H, W]
            feat = feat_S.view(B, Dd, S, H, W).transpose(1, 2).contiguous()
            
        # [B, 3, 198, H, W] 展平为 [B, 594, H, W] 交给最后的大术士
        feat_final = feat.view(B, S * Dd, H, W)
        noise = self.head(feat_final)
        
        # 只取中心切片的原图进行去噪
        center_slice = x[:, 1, :, :, :]
        out = center_slice - noise
        return out
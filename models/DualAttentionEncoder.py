import torch
import torch.nn as nn
import torch.nn.functional as F

num_windows =11
num_channels =14
num_FBs=3
num_classes=3
class Dual_Attention_Encoder(nn.Module):
    def __init__(self, num_windows, num_channels, num_FBs, num_classes,
                 num_heads=8, dim_feedforward=2048, num_layers=6):
        super().__init__()

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=num_channels * num_FBs,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers
        )
        self.temporal_norm = nn.LayerNorm(num_channels * num_FBs)

        self.channel_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=num_windows * num_FBs,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers
        )
        self.channel_norm = nn.LayerNorm(num_windows * num_FBs)

        self.inter_attn = nn.MultiheadAttention(
            embed_dim=num_windows * num_channels * num_FBs,
            num_heads=num_heads,
            batch_first=True
        )
        self.inter_norm = nn.LayerNorm(num_windows * num_channels * num_FBs)

        self.classifier = nn.Sequential(
            nn.Linear(2 * num_windows * num_channels * num_FBs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes))
        self.psd_proj = nn.Linear(num_FBs, num_FBs)

    def forward(self, psd):

        batch = psd.size(0)
        psd = self.psd_proj(psd)  # [B, W, C, F]

        temporal_tokens = psd.reshape(batch, -1, num_channels * num_FBs)  # [B, W, C*F]
        temporal_out = self.temporal_encoder(temporal_tokens)
        temporal_out = self.temporal_norm(temporal_out)  # [B, W, C*F]

        # --- Channel Path ---
        channel_tokens = psd.permute(0, 2, 1, 3).reshape(batch, -1, num_windows * num_FBs)  # [B, C, W*F]
        channel_out = self.channel_encoder(channel_tokens)
        channel_out = self.channel_norm(channel_out)  # [B, C, W*F]

        # Step 3: Inter-Attention交互
        # 拼接时空特征
        temporal_pool = temporal_out.mean(dim=1)  # [B, C*F]
        channel_pool = channel_out.mean(dim=1)  # [B, W*F]
        joint_features = torch.cat([temporal_pool, channel_pool], dim=1)  # [B, (C+W)*F]

        # 交叉注意力计算
        inter_query = joint_features.unsqueeze(1)  # [B, 1, D]
        inter_key = joint_features.unsqueeze(1)
        inter_val = joint_features.unsqueeze(1)
        inter_out, _ = self.inter_attn(inter_query, inter_key, inter_val)
        inter_out = self.inter_norm(inter_out.squeeze(1))  # [B, D]

        # Step 4: 分类输出
        logits = self.classifier(inter_out)
        return logits
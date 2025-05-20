import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from CLDAE.models.DataAugmentation import contrastive_data_augmentation
from CLDAE.models.DualAttentionEncoder import Dual_Attention_Encoder
from CLDAE.modules.PSD import compute_psd
from CLDAE.modules.loss import compute_total_loss


class PreTrainer:
    def __init__(self, model, num_groups, lambda_param=0.5, lr=1e-4):
        self.model = model
        self.num_groups = num_groups  # 每组包含的样本数（N+1）
        self.lambda_param = lambda_param

        # 添加投影头用于对比学习
        self.projection_head = nn.Sequential(
            nn.Linear(2 * model.num_windows * model.num_channels * model.num_FBs, 256),
            nn.ReLU(),
            nn.Linear(256, 128))

        # 冻结分类器，仅训练编码部分
        for param in self.model.classifier.parameters():
            param.requires_grad = False

        self.optimizer = optim.AdamW([
            {'params': self.model.parameters()},
            {'params': self.projection_head.parameters()}
        ], lr=lr)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def _apply_da(self, signals):
        """应用数据增强生成对比样本"""
        # signals: [B, Subjects, C, T, F]
        aug_signals = []
        for batch in signals:
            # 对每个样本应用DA
            augmented = contrastive_data_augmentation(
                batch.numpy(),  # 假设输入为numpy格式
                split_ratio=0.5,
                subjects_per_group=self.num_groups
            )  # [Augmented, C, split_T, F]
            aug_signals.append(torch.from_numpy(augmented))

        # 合并所有增强样本 [Total, C, T, F]
        return torch.cat(aug_signals, dim=0)

    def _compute_loss(self, features):
        """计算对比损失"""
        # 投影到对比空间
        projections = self.projection_head(features)  # [2G, D]

        # 确保输入符合损失函数要求
        G = projections.size(0) // 2
        return compute_total_loss(projections, G, self.lambda_param)

    def train_step(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            raw_signals, _ = batch  # [B, Subjects, C, T, F]

            # 数据增强生成对比样本
            aug_signals = self._apply_da(raw_signals)  # [2G, C, T, F]

            # 提取PSD特征
            psd_features = compute_psd(aug_signals)  # [2G, W, C, F]

            # 前向传播
            features = self.model(psd_features, return_features=True)  # [2G, D]

            # 计算损失
            loss = self._compute_loss(features)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / len(dataloader)



# 使用示例
if __name__ == "__main__":
    # 参数配置
    num_windows = 11  # 时间窗数量W
    num_channels = 14  # EEG通道数C
    num_FBs = 3  # 频带数F
    num_groups = 2

    model = Dual_Attention_Encoder(num_windows, num_channels, num_FBs, num_classes=4)
    trainer = PreTrainer(model, num_groups, lambda_param=0.7)

    # 模拟数据加载器
    dummy_data = torch.randn(32, 3, 14, 128, 5)  # [B, Subjects, C, T, F]
    dummy_labels = torch.randint(0, 4, (32,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=8)

    # 预训练循环
    for epoch in range(100):
        loss = trainer.train_step(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
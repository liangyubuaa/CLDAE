import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score


class FineTuner:
    def __init__(self, pretrained_model, num_classes, freeze_encoder=True, lr=1e-3):
        """
        Args:
            pretrained_model: 预训练好的Dual_Attention_Encoder模型
            num_classes: 下游任务类别数
            freeze_encoder: 是否冻结编码器参数
            lr: 微调学习率
        """
        self.model = pretrained_model

        # 替换分类器适配下游任务
        self.model.classifier = nn.Sequential(
            nn.Linear(2 * self.model.num_windows * self.model.num_channels * self.model.num_FBs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # 冻结编码器参数
        if freeze_encoder:
            for name, param in self.model.named_parameters():
                if 'classifier' not in name:  # 只训练分类器
                    param.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=3)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in train_loader:
            psd_features, labels = batch  # 假设数据已经过PSD预处理

            self.optimizer.zero_grad()

            # 前向传播
            logits = self.model(psd_features)
            loss = self.criterion(logits, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 记录指标
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return total_loss / len(train_loader), acc, f1

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                psd_features, labels = batch

                logits = self.model(psd_features)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        self.scheduler.step(acc)  # 根据验证指标调整学习率
        return total_loss / len(val_loader)), acc, f1

    def fit(self, train_loader, val_loader, epochs=50):
        best_acc = 0
        for epoch in range(epochs):
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1 = self.validate(val_loader)

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), f'best_model_epoch{epoch + 1}.pth')

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%} | F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.4f}\n")


# 使用示例
if __name__ == "__main__":
    # 加载预训练模型
    pretrained_model = Dual_Attention_Encoder(
        num_windows=10,
        num_channels=14,
        num_FBs=5,
        num_classes=4  # 预训练时使用的虚拟类别数
    )
    pretrained_model.load_state_dict(torch.load('pretrained.pth'))

    # 初始化微调器
    finetuner = FineTuner(
        pretrained_model,
        num_classes=3,  # MAN数据集3分类
        freeze_encoder=True,  # 第一阶段冻结编码器
        lr=1e-3
    )

    # 加载微调数据集（需要实现数据管道）
    train_loader = DataLoader(...)  # 需返回(psd_features, labels)
    val_loader = DataLoader(...)

    # 开始微调
    finetuner.fit(train_loader, val_loader, epochs=60)
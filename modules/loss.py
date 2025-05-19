import torch
import torch.nn as nn


def compute_total_loss(C, G, lambda_param):
    """
    C: 模型输出的特征张量，形状为 [2*G, D]，其中每两个连续样本属于一个组
    G: 组数
    lambda_param: 公式中的λ参数，控制组间对比损失的权重
    """
    # 确保输入的形状正确
    assert C.shape[0] == 2 * G, "输入特征的数量必须是2倍的组数"

    # 将特征划分为G组，每组两个样本
    group_features = C.view(G, 2, -1)  # 形状 [G, 2, D]

    # 计算组合损失 L_combined: sum_{k=1}^G MSE(C_{2k-1}, C_{2k})
    pair_mse = torch.mean((group_features[:, 0] - group_features[:, 1]) ** 2, dim=1)
    L_combined = torch.sum(pair_mse)

    # 计算每个组的平均特征 O_g
    O = torch.mean(group_features, dim=1)  # 形状 [G, D]

    # 计算组间对比损失 L_group_contrastive: sum_{i<j} MSE(O_i, O_j)
    diff = O.unsqueeze(1) - O.unsqueeze(0)  # 形状 [G, G, D]
    mse_matrix = torch.mean(diff ** 2, dim=2)  # 形状 [G, G]
    triu_indices = torch.triu_indices(row=G, col=G, offset=1)
    L_group_contrastive = torch.sum(mse_matrix[triu_indices[0], triu_indices[1]])

    # 总损失
    L_total = L_combined - lambda_param * L_group_contrastive

    return L_total
import numpy as np
from itertools import combinations


def contrastive_data_augmentation(signal, split_ratio=0.5, subjects_per_group=2):

    num_subjects, num_channels, total_time, num_features = signal.shape
    split_point = int(total_time * split_ratio)

    augmented_pairs = []

    # 步骤1: 按公式(3)生成所有被试组合 (C(N+1,2))
    subject_groups = combinations(range(num_subjects), subjects_per_group)

    for group in subject_groups:
        # 每组随机选择两个被试 (si, sj)
        for si, sj in combinations(group, 2):
            # 公式(5)-(6): 按split_point分割并交叉重组
            si_front = signal[si, :, :split_point, :]  # 维度 [C, split_T, F]
            si_rear = signal[si, :, split_point:, :]

            sj_front = signal[sj, :, :split_point, :]
            sj_rear = signal[sj, :, split_point:, :]

            # 生成两种重组方式
            xi = np.concatenate([si_front, sj_rear], axis=1)  # 沿时间轴拼接
            xj = np.concatenate([sj_front, si_rear], axis=1)

            augmented_pairs.extend([xi, xj])

    augmented_signals = np.stack(augmented_pairs, axis=0)

    return augmented_signals
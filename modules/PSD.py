import numpy as np
import mne
from scipy import signal
from mne.preprocessing import ICA

sfreq=128
window_duration=0.25
segment_duration=3
def compute_psd(aug_signals):
    """
    计算增强后EEG信号的功率谱密度(PSD)特征

    参数：
    aug_signals : np.ndarray
        增强后的EEG信号，形状为 [样本数, 通道数, 时间点]
    sfreq : int
        采样频率 (默认128Hz)
    window_duration : float
        PSD计算窗口时长 (秒) (默认0.25秒)
    segment_duration : int
        数据分段时长 (秒) (默认3秒)

    返回：
    PSDs : np.ndarray
        PSD特征矩阵，形状为 [总窗口数, 通道数, 频带数]
    """

    # 参数计算
    num_samples, num_channels, num_timepoints = aug_signals.shape
    window_size = int(sfreq * window_duration)
    hop_size = window_size  # 无重叠
    segment_length = int(sfreq * segment_duration)

    # 定义频带
    freq_bands = {
        'theta': (4, 8),
        'alpha': (8, 14),
        'beta': (14, 31)
    }
    num_freq_bands = len(freq_bands)

    # 创建MNE原始数据对象
    ch_names = [f'EEG{i}' for i in range(num_channels)]
    ch_types = ['eeg'] * num_channels
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    # 预处理管道
    all_PSDs = []

    for sample in aug_signals:
        # 转换为MNE格式 [通道数, 时间点]
        raw = mne.io.RawArray(sample, info=info)

        # 带通滤波
        raw.filter(l_freq=1.0, h_freq=50.0, verbose=False)

        # ICA处理 (可选)
        # ica = ICA(n_components=num_channels, max_iter=1000, random_state=0)
        # ica.fit(raw, verbose=False)
        # ica.apply(raw, verbose=False)

        # 获取滤波后数据
        filtered_data = raw.get_data()  # [通道数, 时间点]

        # 划分3秒片段
        num_segments = filtered_data.shape[1] // segment_length
        PSDs = []

        for seg_idx in range(num_segments):
            start = seg_idx * segment_length
            end = start + segment_length
            segment = filtered_data[:, start:end]

            # 滑动窗口处理
            num_windows = (segment_length - window_size) // hop_size + 1

            for win_idx in range(num_windows):
                win_start = win_idx * hop_size
                win_end = win_start + window_size
                window_data = segment[:, win_start:win_end]

                # 计算Welch PSD
                freqs, psd = signal.welch(
                    window_data,
                    fs=sfreq,
                    nperseg=window_size,
                    axis=1,
                    scaling='density',
                    average='mean'
                )

                # 提取频带能量
                band_energies = np.zeros((num_channels, num_freq_bands))

                for ch in range(num_channels):
                    for band_idx, (f_low, f_high) in enumerate(freq_bands.values()):
                        band_mask = (freqs >= f_low) & (freqs <= f_high)
                        band_energies[ch, band_idx] = np.sum(psd[ch][band_mask])

                PSDs.append(band_energies)

        all_PSDs.extend(PSDs)

    # 转换为三维数组 [总窗口数, 通道数, 频带数]
    return np.array(all_PSDs)
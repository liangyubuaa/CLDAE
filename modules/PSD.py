import os
import glob
import numpy as np
import pandas as pd
from scipy import signal
import mne
from mne.preprocessing import ICA

data_folder = ''  # define the path of the data folder
data_files = glob.glob(os.path.join(data_folder, '*.csv'))

for data_file in data_files:
    # load EEG data
    data = pd.read_csv(data_file, header=0)
    eeg_channels = data.columns[1:15]
    eeg_data = data[eeg_channels].values
    sfreq = 128  # sampling frequency is 128 Hz

    # crate MNE raw object
    ch_names = list(eeg_channels)
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(eeg_data.T, info=info)

    # bandpass filter
    raw.filter(l_freq=1.0, h_freq=50.0)

    # create ICA object and fit it to raw data
    ica = ICA(n_components=len(eeg_channels), random_state=0, max_iter=1000)  # 调整参数
    ica.fit(raw)

    # apply ICA to raw data
    ica.exclude = []
    ica.apply(raw)

    # get filtered data
    filtered_data = raw.get_data()

    # define frequency bands
    freq_bands = {
        # 'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 14),
        'beta': (14, 31),
        # 'gamma': (31, 50),
    }

    # parameters for PSD extraction
    window_size = int(sfreq * 0.25)  # window size is 0.25 seconds
    overlap = 0.0  # overlap between consecutive windows is 0.0
    hop_size = int(window_size * (1 - overlap))

    # one sample is 3 seconds data
    segment_duration = 3 * sfreq
    num_segments = int(np.floor(filtered_data.shape[1] / segment_duration))

    num_channels = len(eeg_channels)
    num_windows_per_segment = int(np.floor((segment_duration - window_size) / hop_size)) + 1
    num_total_windows = num_windows_per_segment * num_segments
    print(num_total_windows)

    num_freq_bands = len(freq_bands)
    psd_results = np.zeros((num_total_windows, num_channels * num_freq_bands))

    # extract PSD features
    window_idx = 0
    for segment_idx in range(num_segments):
        start = segment_idx * segment_duration
        end = start + segment_duration
        segment_data = filtered_data[:, start:end]

        for window_start in range(0, segment_duration - window_size + 1, hop_size):
            window_end = window_start + window_size
            eeg_window = segment_data[:, window_start:window_end]
            # welch PSD
            frequencies, psd = signal.welch(eeg_window, fs=sfreq, nperseg=window_size)
            for channel_idx in range(num_channels):
                for band_idx, (band_name, (f_low, f_high)) in enumerate(freq_bands.items()):
                    band_mask = np.logical_and(frequencies >= f_low, frequencies <= f_high)
                    band_psd = psd[channel_idx, band_mask]
                    PSDs[window_idx, channel_idx * num_freq_bands + band_idx] = np.sum(band_psd)
                    PSDs = PSDs.reshape(num_total_windows, num_channels, num_freq_bands)
            window_idx += 1

    # PSDs : (num_total_windows, num_channels, num_fre q_bands)
    print(PSDs.shape)

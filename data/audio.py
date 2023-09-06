import os
import sys

import librosa
import numpy as np
import wave

from utils import get_sorted_files

def analyze_audio_feature(root_path):
    merged_audio = []
    files = get_sorted_files(root_path, ".wav")
    for name in files:
        audio_file_path = os.path.join(root_path, name)
        wavefile = wave.open(audio_file_path)
        audio_sr = wavefile.getframerate()
        n_samples = wavefile.getnframes()
        signal = np.frombuffer(wavefile.readframes(n_samples), dtype=np.short)
        signal = signal.astype(float)
        merged_audio = np.hstack((merged_audio, signal))
    mel_spectro = normalize(convert_mel_spectrogram(merged_audio, audio_sr))
    return mel_spectro

def convert_mel_spectrogram(audio, audio_sr, frame_size=2048, hop_size=533, num_mel_bands=80):
    mel_spectrogram = librosa.feature.melspectrogram(audio,
                                                     sr=audio_sr,
                                                     n_fft=frame_size,
                                                     hop_length=hop_size,
                                                     n_mels=num_mel_bands)
    # convert amplitude to DBs
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram  # in dB

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std



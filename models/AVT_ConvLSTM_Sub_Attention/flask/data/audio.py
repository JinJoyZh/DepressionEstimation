import os

import librosa
import numpy as np
import wave

def analyze_audio_feature(root_path):
    global audio_sr
    dir_tree = os.walk(root_path)
    merged_audio = []
    for root, dirs, files in dir_tree:
        for name in files:
            if name.endwith(".wav"):
                audio_file_path = os.path.join(root, name)
                wavefile = wave.open(audio_file_path)
                audio_sr = wavefile.getframerate()
                n_samples = wavefile.getnframes()
                signal = np.frombuffer(wavefile.readframes(n_samples), dtype=np.short)
                signal = signal.astype(float)
                merged_audio = np.hstack((merged_audio, signal))
    mel_spectro = normalize(convert_mel_spectrogram(merged_audio, audio_sr, frame_size=2048, hop_size=533, num_mel_bands=80))
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



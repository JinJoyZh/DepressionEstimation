import os

import numpy as np

WINDOW_SIZE     = 15    # 60s
OVERLAP_SIZE    = 2    # 10s

def sliding_window(fkps_features, gaze_features, mel_spectro, text_feature, visual_sr):
    frame_size = WINDOW_SIZE * visual_sr
    hop_size = (WINDOW_SIZE - OVERLAP_SIZE) * visual_sr
    num_frame = get_num_frame(fkps_features, frame_size, hop_size)
    text_frame_size = 10
    text_hop_size = get_text_hop_size(text_feature, text_frame_size, num_frame)

    # start sliding through and generating data
    for i in range(num_frame):
        frame_sample_fkps = visual_padding(fkps_features[i * hop_size:i * hop_size + frame_size], frame_size)
        frame_sample_gaze = visual_padding(gaze_features[i * hop_size:i * hop_size + frame_size], frame_size)
        frame_sample_mspec = audio_padding(mel_spectro[:, i * hop_size:i * hop_size + frame_size], frame_size)
        frame_sample_text = text_padding(text_feature[i * text_hop_size:i * text_hop_size + text_frame_size],
                                         text_frame_size)

    return frame_sample_fkps, frame_sample_gaze, frame_sample_mspec, frame_sample_text

def get_text_hop_size(text, frame_size, num_frame):
    T = text.shape[0]
    return (T - frame_size) // (num_frame - 1)

def get_num_frame(data, frame_size, hop_size):
    T = data.shape[0]
    if (T - frame_size) % hop_size == 0:
        num_frame = (T - frame_size) // hop_size + 1
    else:
        num_frame = (T - frame_size) // hop_size + 2
    return num_frame

def visual_padding(data, pad_size):
    if data.shape[0] != pad_size:
        size = tuple()
        size = size + (pad_size,) + data.shape[1:]
        padded_data = np.zeros(size)
        padded_data[:data.shape[0]] = data
    else:
        padded_data = data

    return padded_data

def audio_padding(data, pad_size):
    if data.shape[1] != pad_size:
        size = tuple((data.shape[0], pad_size))
        padded_data = np.zeros(size)
        padded_data[:, :data.shape[1]] = data
    else:
        padded_data = data

    return padded_data

def text_padding(data, pad_size):
    if data.shape[0] != pad_size:
        size = tuple((pad_size, data.shape[1]))
        padded_data = np.zeros(size)
        padded_data[:data.shape[0]] = data
    else:
        padded_data = data

    return padded_data
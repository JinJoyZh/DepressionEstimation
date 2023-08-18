
import os
import re
import shutil
import sys
from flask import Flask
import numpy as np

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)

from data.video import analyze_video_feature


# 访谈记录数据存放格式
# ├─interviewer_ID_timezone     //单次访谈的根目录存放根目录
#     ├─transcript.csv          //仅存放受访者回答的文字记录
#     ├─video_timezone.wmv      //受访者单次说话的录像
#     ├─audio_timezone.wav      //受访者单次说话的录音

# 缓存存放目录
# AVT_ConvLSTM_Sub_Attention/flask/cache/1234


class MultimodalDiagnosticService:

    lastest_video_file_num = 0

    def __init__(self, user_data_dir):
        self.USER_DATA_DIR = user_data_dir
        self.USER_ID = user_data_dir.split("_")[-2]
        cache_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(cache_dir, "cache", self.USER_ID)
        self.CACHE_DIR = cache_dir
        self.VIDEO_CACHE = os.path.join(self.CACHE_DIR, "video")
        self.TEXT_CACHE = os.path.join(self.CACHE_DIR, "text")
        self.AUDIO_CACHE = os.path.join(self.CACHE_DIR, "audio")
        # shutil.rmtree(cache_dir)
        self._make_dirs([self.VIDEO_CACHE, self.AUDIO_CACHE, self.TEXT_CACHE])

    def _make_dirs(self, dirs):
        for dir in dirs:
            if not os.path.exists(dir) :
                os.makedirs(dir) 

    def generate_video_features(self, input_path, skip_frame = 0):
        time_zone = int(re.findall(r"\d+",input_path)[-1])
        cache_dir = os.path.join(self.VIDEO_CACHE, time_zone)
        key_points_set, gaze_set = analyze_video_feature(input_path, cache_dir, skip_frame)
        key_point_feature_path = os.path.join(self.VIDEO_CACHE, "video" + time_zone + ".npy")
        np.save(key_points_set, key_point_feature_path)
        gaze_feature_path = os.path.join(self.VIDEO_CACHE, "gaze_" + time_zone + ".npy")
        np.save(gaze_set, gaze_feature_path)

    def generate_audio_features(self, )

    # def generate_text_features(self, transcrpit_file_path):

    

if __name__ == '__main__':
        # user_data_dir = "aas_1234_as"
        # USER_DATA_DIR = user_data_dir
        # USER_ID = user_data_dir.split("_")[-2]
        # cache_dir = os.path.dirname(os.path.abspath(__file__))
        # cache_dir = os.path.join(cache_dir, "cache", USER_ID)
        # print(cache_dir)
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    print(root_dir)

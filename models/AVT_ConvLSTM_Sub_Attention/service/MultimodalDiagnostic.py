
import os
import re
import shutil
import sys
import time
import numpy as np
import pandas as pd
from autolab_core import YamlConfig
import torch

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)

from data.audio import analyze_audio_feature
from data.text import analyze_text_feature
from data.video import analyze_video_feature
from data.sliding import sliding_window
from inference import model_processing
from utils import compute_score, get_sorted_files, init_seed

# 访谈记录数据存放格式
# ├─interviewer_ID_timezone     //单次访谈的根目录存放根目录
#     ├─transcript.csv          //仅存放受访者回答的文字记录
#     ├─video_timezone.wmv      //受访者单次说话的录像
#     ├─audio_timezone.wav      //受访者单次说话的录音

# 缓存目录
# ├── cache
# │   └── user_id
# │       └── video
# │           ├── gaze
# │           └── keypoints

class MultimodalDiagnostic:

    lastest_video_file_num = 0

    # user_data_dir 存放访谈记录数据的绝对路径
    def __init__(self, user_data_dir):
        self.USER_DATA_DIR = user_data_dir
        self.USER_ID = user_data_dir.split("_")[-2]
        self.CACHE_DIR = os.path.join(user_data_dir, "cache")
        self.VIDEO_CACHE = os.path.join(self.CACHE_DIR, "video")
        self.GAZE_CACHE = os.path.join(self.VIDEO_CACHE, "gaze")
        self.KEYPOINT_CACHE = os.path.join(self.VIDEO_CACHE, "keypoints")
        self._make_dirs([self.VIDEO_CACHE, self.GAZE_CACHE, self.KEYPOINT_CACHE])

    def _make_dirs(self, dirs):
        for dir in dirs:
            if not os.path.exists(dir) :
                os.makedirs(dir) 

    #ivideo_file_name 视频文件全名。视频文件格式支持mp4, wmv
    #skip_frame 分析视频过程中跳帧的数量。可以增加改数值减小算力消耗，但不推荐这么做
    def generate_video_features(self, video_file_name, skip_frame = '0'):
        input_path = os.path.join(self.USER_DATA_DIR, video_file_name)
        time_zone = int(re.findall(r"\d+",input_path)[-1])
        time_zone = str(time_zone)
        cache_dir = os.path.join(self.VIDEO_CACHE, time_zone)
        key_points_set, gaze_set = analyze_video_feature(input_path, cache_dir, skip_frame)
        key_point_feature_path = os.path.join(self.KEYPOINT_CACHE, "video_" + time_zone + ".npy")
        np.save(key_point_feature_path, key_points_set)
        gaze_feature_path = os.path.join(self.GAZE_CACHE, "gaze_" + time_zone + ".npy")
        np.save(gaze_feature_path, gaze_set)

    def _generate_audio_features(self, root_path):
        return analyze_audio_feature(root_path)
    
    def _generate_text_features(self, transcrpit_file_path):
        return analyze_text_feature(transcrpit_file_path)
    
    # @Param 
    # visual_sr 视频帧率
    # @return
    # probs     PHQ值
    def generate_phq(self, visual_sr):
        print("The Multimodal analysis starts!")
        start = time.time()
        config_file = os.path.join(root_dir, 'config/config_phq-subscores.yaml')
        config = YamlConfig(config_file)
        phq_score_pred = []
        phq_binary_pred = []
        # set up torch device: 'cpu' or 'cuda' (GPU)
        args = self.Args()
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.gpu = '0'
        # create the output folder (name of experiment) for storing model result such as logger information
        if not os.path.exists(config['OUTPUT_DIR']):
            os.mkdir(config['OUTPUT_DIR'])
        # print configuration
        print('=' * 40)
        print(config.file_contents)
        config.save(os.path.join(config['OUTPUT_DIR'], config['SAVE_CONFIG_NAME']))
        print('=' * 40)
        # initialize random seed for torch and numpy
        init_seed(config['MANUAL_SEED'])
        # load audio feature
        mel_spectro = self._generate_audio_features(self.USER_DATA_DIR)
        # load text feature
        transcrpit_file_name = "transcript.csv"
        transcrpit_file_path = os.path.join(self.USER_DATA_DIR, transcrpit_file_name)
        text_features = self._generate_text_features(transcrpit_file_path)
        # load video feature
        gaze_features = self._load_video_feature(self.GAZE_CACHE)
        keypoint_features = self._load_video_feature(self.KEYPOINT_CACHE)
        frame_sample_fkps, frame_sample_gaze, frame_sample_mspec, frame_sample_text = sliding_window(keypoint_features, gaze_features, mel_spectro, text_features, visual_sr)
        visual_feature = np.concatenate((frame_sample_fkps, frame_sample_gaze), axis=1)
        visual_feature = torch.from_numpy(np.asarray([visual_feature], dtype='float32'))
        frame_sample_mspec = torch.from_numpy(np.asarray([frame_sample_mspec], dtype='float32'))
        frame_sample_text = torch.from_numpy(np.asarray([frame_sample_text], dtype='float32'))
        input = {'visual': visual_feature, 'audio': frame_sample_mspec, 'text': frame_sample_text}
        probs = model_processing(input, config, args)
        # predict the final score
        config = config['MODEL']
        pred_score = compute_score(probs, config['EVALUATOR'], args)
        print("pred_score: " + str(len(pred_score)))
        phq_score_pred.extend([pred_score[i].item() for i in range(1)])  # 1D list
        print("phq_score_pred: " + str(phq_score_pred))
        phq_binary_pred.extend([1 if pred_score[i].item() >= config['PHQ_THRESHOLD'] else 0 for i in range(1)])
        print("phq_binary_pred: " + str(phq_binary_pred))
        end = time.time()
        time_cost = str(end - start)
        print("Multimodal analysis ends. Time cost: " + time_cost + " seconds")
        #delete cache
        shutil.rmtree(self.CACHE_DIR)
        return phq_score_pred[0], phq_binary_pred[0]

    def _load_video_feature(self, root_dir):
        files = get_sorted_files(root_dir, ".npy")
        features = np.asarray([])
        for file in files:
            path = os.path.join(root_dir, file)
            data = np.load(path)
            if features.shape == (0, ):
                features = data
            else:
                features = np.append(features, data, axis = 0)
        return features

    # 添加新的访谈文字记录 
    # start_time 受访者开始说这句话的时间
    # end_time  受访者结束说这句话的时间
    # value     文本内容（英文）   
    def transcript(self, start_time, end_time, value):
        file_path = os.path.join(self.USER_DATA_DIR, 'transcript.csv')
        city = pd.DataFrame([[start_time, end_time, value]], columns=['start_time', 'end_time', 'value'])
        need_header = True
        if os.path.isfile(file_path):
            need_header = False
        city.to_csv(file_path, mode = 'a', header = need_header)
    
    class Args(object):
        pass
    
if __name__ == '__main__':
    user_data_dir = "/home/zjy/workspace/tmp/interviewee_12345_1692723605"
    serivce = MultimodalDiagnostic(user_data_dir)
    print('start to get video feature')
    serivce.generate_video_features("vdieo_1692758150.wmv")
    serivce.generate_video_features("video_1692757670.wmv")
    serivce.generate_video_features("video_1692757850.wmv")
    video_frame_rate = 30
    phq_score_pred, phq_binary_pred = serivce.generate_phq(30)
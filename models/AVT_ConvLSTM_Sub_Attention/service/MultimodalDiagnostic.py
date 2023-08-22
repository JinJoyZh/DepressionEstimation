
import os
import re
import shutil
import sys
import numpy as np
from autolab_core import YamlConfig

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)

from utils import get_sorted_files
from data.audio import analyze_audio_feature
from data.text import analyze_text_feature
from data.video import analyze_video_feature
from data.sliding import sliding_window
from utils import init_seed
from inference import model_processing

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

    def __init__(self, user_data_dir):
        self.USER_DATA_DIR = user_data_dir
        self.USER_ID = user_data_dir.split("_")[-2]
        cache_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(cache_dir, "cache", self.USER_ID)
        self.CACHE_DIR = cache_dir
        self.VIDEO_CACHE = os.path.join(self.CACHE_DIR, "video")
        self.GAZE_CACHE = os.path.join(self.VIDEO_CACHE, "gaze")
        self.KEYPOINT_CACHE = os.path.join(self.VIDEO_CACHE, "keypoints")
        self._make_dirs([self.VIDEO_CACHE, self.GAZE_CACHE, self.KEYPOINT_CACHE])

    def _make_dirs(self, dirs):
        for dir in dirs:
            if not os.path.exists(dir) :
                os.makedirs(dir) 

    #input_path 视频文件的绝对路径。视频文件格式支持mp4, wmv
    #skip_frame 分析视频过程中跳帧的数量。可以增加改数值减小算力消耗，但不推荐这么做
    def generate_video_features(self, input_path, skip_frame = 0):
        time_zone = int(re.findall(r"\d+",input_path)[-1])
        time_zone = str(time_zone)
        cache_dir = os.path.join(self.VIDEO_CACHE, time_zone)
        key_points_set, gaze_set = analyze_video_feature(input_path, cache_dir, skip_frame)
        key_point_feature_path = os.path.join(self.KEYPOINT_CACHE, "video" + time_zone + ".npy")
        np.save(key_points_set, key_point_feature_path)
        gaze_feature_path = os.path.join(self.GAZE_CACHE, "gaze_" + time_zone + ".npy")
        np.save(gaze_set, gaze_feature_path)

    def _generate_audio_features(self, root_path):
        return analyze_audio_feature(root_path)
    
    def _generate_text_features(self, transcrpit_file_path):
        return analyze_text_feature(transcrpit_file_path)
    
    # @Param 
    # root_path 存放用户采访数据的根目录interviewer_ID_timezone的绝对路径
    # visual_sr 视频帧率
    # @return
    # probs     PHQ值
    def generate_phq(self, root_path, visual_sr):
        # set up torch device: 'cpu' or 'cuda' (GPU)
        args = []
        args.device = 'cuda'
        config_file = '../config/config_phq-subscores.yaml'
        config = YamlConfig(config_file)
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
        mel_spectro = self._generate_audio_features(root_path)
        # load text feature
        transcrpit_file_name = "transcript.csv"
        transcrpit_file_path = os.path.join(root_path, transcrpit_file_name)
        text_features = self._generate_text_features(transcrpit_file_path)
        # load video feature
        gaze_features = self._load_video_feature(self.GAZE_CACHE)
        keypoint_features = self._load_video_feature(self.KEYPOINT_CACHE)
        frame_sample_fkps, frame_sample_gaze, frame_sample_mspec, frame_sample_text = sliding_window(keypoint_features, gaze_features, mel_spectro, text_features, visual_sr)
        visual_feature = np.concatenate((frame_sample_fkps, frame_sample_gaze), axis=1)
        input = {'visual': visual_feature, 'audio': frame_sample_mspec, 'text': frame_sample_text}
        probs = model_processing(input, config, args)
        #delete cache
        shutil.rmtree(self.CACHE_DIR)
        return probs


    def _load_video_feature(root_dir):
        files = get_sorted_files(root_dir, ".npy")
        features = np.asarray([])
        for file in files:
            data = np.load(file)
            if features.shape == (0, ):
                features = data
            else:
                features = np.append(features, data, axis = 0)
        return features
        


if __name__ == '__main__':
    # user_data_dir = "aas_1234_as"
    # USER_DATA_DIR = user_data_dir
    # USER_ID = user_data_dir.split("_")[-2]
    # cache_dir = os.path.dirname(os.path.abspath(__file__))
    # cache_dir = os.path.join(cache_dir, "cache", USER_ID)
    # print(cache_dir)
    # root_dir = os.path.abspath(os.path.dirname(__file__))
    # print(root_dir)
    a = np.load("/home/zjy/workspace/DepressionRec/dataset/DAIC_WOZ-generated_database_V2/test/clipped_data/gaze_vectors/300-00_gaze.npy")
    print(a.shape)

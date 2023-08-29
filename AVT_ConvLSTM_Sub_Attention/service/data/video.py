import os
import shutil
import sys
import cv2
import time
import numpy as np
import pandas as pd

root_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.join(root_dir, "../.."))
sys.path.append(root_dir)

from utils import get_sorted_files

KEY_POINT_NUM = 68

def analyze_video_feature(input_dir, output_dir, skip_frame ='0'):
    # limit OpenBlas to accelerate in single thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    exe_path = os.path.join(root_dir, "bin", "FaceAnalyzerVid")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
    # extract feature with OpenFace
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cmd = exe_path + " -f " + input_dir + " -out_dir " + output_dir +  " -skip_frame " + skip_frame
    os.system(cmd)
    key_points_set, gaze_set = record_video_feature(output_dir)
    return key_points_set, gaze_set
    
def record_video_feature(csv_file_dir):
    print('record_video_feature start')
    start = time.time()
    # reshape video feature
    gaze_set = []
    key_points_set = []
    csv_files = get_sorted_files(csv_file_dir, ".csv")
    for csv_file in csv_files:
        gaze_path = csv_file_dir + "/" + csv_file
        gaze_df = pre_check(pd.read_csv(gaze_path, low_memory=False))
        gaze_coor = gaze_df.iloc[:, 2:8].to_numpy().reshape(2, 3) 
        # Calculate euler angle
        rotation_vec = gaze_df.iloc[:, 10:13].to_numpy().reshape(1, 3) 
        translation_vec = gaze_df.iloc[:, 13:16].to_numpy().reshape(1, 3) 
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec.T))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        # get gaze vector in head coordinate space
        rotation_mat = eul2rot(euler_angles)
        rotation_mat = np.asarray(rotation_mat)
        rotation_mat = rotation_mat.reshape(3, 3)
        gaze_coor_h = np.dot(rotation_mat, gaze_coor[0])
        gaze_coor = np.vstack((gaze_coor, gaze_coor_h))
        gaze_coor_h = np.dot(rotation_mat, gaze_coor[1])
        gaze_coor = np.vstack((gaze_coor, gaze_coor_h))
        gaze_set.append(gaze_coor)
        #get key points
        coors = np.asarray([])
        for i in range(0, KEY_POINT_NUM):
            x = gaze_df.iloc[:, 16 + i].to_numpy()
            y = gaze_df.iloc[:, 16 + i + KEY_POINT_NUM].to_numpy()
            z = gaze_df.iloc[:, 16 + i + KEY_POINT_NUM * 2].to_numpy()
            coor = [x, y, z]
            coor = np.asarray(coor)
            coor = coor.reshape(1,3)
            if(coors.shape == (0,)):
                coors = coor
            else:
                coors = np.vstack((coors, coor))
        key_points_set.append(coors)
    gaze_set = np.asarray(gaze_set)
    key_points_set = np.asarray(key_points_set)
    end = time.time()
    print('record_video_feature end, time cost ' + str(end - start) + " seconds")
    return key_points_set, gaze_set

def pre_check(data_df):
    data_df = data_df.apply(pd.to_numeric, errors='coerce')
    data_np = data_df.to_numpy()
    data_min = data_np[np.where(~(np.isnan(data_np[:, 4:])))].min()
    data_df.where(~(np.isnan(data_df)), data_min, inplace=True)
    return data_df

def eul2rot(theta) :
    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])
    return R


if __name__ == '__main__':
    #In order to prevent memory overflow, some functions of OpenBlus are restricted
    os.system("export OMP_NUM_THREADS=1")
    os.system("export VECLIB_MAXIMUM_THREADS=1")

    # analyze images
    # exe_path =  "/home/zjy/workspace/DepressionRec/alg/OpenFace/experiments/lib/FaceAnalyzerImgs"
    # input_dir = "/home/zjy/workspace/DepressionRec/alg/OpenFace/samples"
    # output_dir = "/home/zjy/workspace/DepressionRec/alg/OpenFace/experiments/output/Imgs"
    # subprocess.Popen([exe_path, "-fdir", input_dir, "-out_dir", output_dir])

    # analyze videos
    # exe_path = "./models/AVT_ConvLSTM_Sub_Attention/flask/bin/FaceAnalyzerVid"
    # exe = os.path.abspath(exe_path)
    # input_dir = "/home/zjy/workspace/DepressionRec/alg/OpenFace/samples/default.wmv"
    # output_dir = "/home/zjy/workspace/DepressionRec/alg/DepressionEstimation/models/AVT_ConvLSTM_Sub_Attention/flask/cache/video_feature"
    # subprocess.Popen([exe, "-f", input_dir, "-out_dir", output_dir, "-skip_frame", "2"])

    # current_directory = os.path.dirname(os.path.abspath(__file__))
    # print(current_directory)
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # path = os.path.join(root_dir, "bin", "FaceAnalyzerVid")
    print(root_dir)
import numpy as np
import pandas as pd



def pre_check(data_df):
    data_df = data_df.apply(pd.to_numeric, errors='coerce')
    data_np = data_df.to_numpy()
    data_min = data_np[np.where(~(np.isnan(data_np[:, 4:])))].min()
    data_df.where(~(np.isnan(data_df)), data_min, inplace=True)
    return data_df


def load_gaze(gaze_path):
    gaze_df = pre_check(pd.read_csv(gaze_path, low_memory=False))
    # process into format TxVxC
    tmp = gaze_df.iloc[:, 4:].to_numpy()

    gaze_coor = gaze_df.iloc[:, 4:].to_numpy().reshape(len(gaze_df), 4, 3)  # 4 gaze vectors, 3 axes
    T, V, C = gaze_coor.shape
    return gaze_coor

def load_keypoints(keypoints_path):
    fkps_df = pre_check(pd.read_csv(keypoints_path, low_memory=False))
    # process into format TxVxC
    x_coor = min_max_scaler(fkps_df[fkps_df.columns[4: 72]].to_numpy())
    y_coor = min_max_scaler(fkps_df[fkps_df.columns[72: 140]].to_numpy())
    z_coor = min_max_scaler(fkps_df[fkps_df.columns[140: 208]].to_numpy())
    fkps_coor = np.stack([x_coor, y_coor, z_coor], axis=-1)
    T, V, C = fkps_coor.shape
    print(fkps_coor.shape)
    print(fkps_coor)

    return fkps_coor

def min_max_scaler(data):
    '''recale the data, which is a 2D matrix, to 0-1'''
    return (data - data.min())/(data.max() - data.min())


if __name__ == '__main__':
    path = "/Users/jinjoy/resource/DepressionRec/dataset/DAIC-WOZ_Database/492_P/492_CLNF_features3D.txt"
    load_keypoints(path)

    # path = "/Users/jinjoy/resource/DepressionRec/dataset/DAIC-WOZ_Database/492_P/492_CLNF_gaze.txt"
    # load_gaze(path)

'''
Aditya Iyer 07 November 2022
Converting annotations on the Wacom Tablet to DeepLabCut csv and panda files
'''
import numpy as np
import pandas as pd
import os, cv2

# path_ = '/home/iyer_la/Documents/tadpole_stage57_Cam2-Aditya-2022-11-04/labeled-data/1_0-1477/'
# df_ = pd.read_hdf(f'{path_}CollectedData_Aditya.h5')
# slice_ = np.linspace(0, 99, 10, dtype = np.int64)
# img_names = [i[-1][:-4] for i in df_.index]
# for i, img_name in enumerate(img_names):
#     tail_pts = np.load(f'{path_}Camera1{img_name}_pts00000.npy')
#     tail_pts = tail_pts[slice_]
#     for j, pt in enumerate(tail_pts):
#         df_[('Aditya', f'tail{j}')].iloc[i] = pt
# df_.to_hdf(f'{path_}CollectedData_Aditya.h5', 'df_with_missing')
# df_.to_csv(f'{path_}CollectedData_Aditya.csv')

def add_tail(folder_name):
    df = pd.read_hdf(f'/home/iyer_la/Documents/tadpole_stage57_Cam2-Aditya-2022-11-04/labeled-data/{folder_name}/CollectedData_Aditya.h5')
    df = df.apply(pd.to_numeric)
    np_path = f'/home/iyer_la/Documents/TailAnnotated/tadpole_stage57_Cam2-Aditya-2022-11-04/labeled-data/{folder_name}/'
    tail_paths = [i for i in os.listdir(np_path) if 'npy' in i]
    img_names = [f'{i[7:14]}.png' for i in tail_paths] #Camera0_img00001
    slice_ = np.linspace(0, 99, 10, dtype = np.int64)
    for i in range(len(tail_paths)):
        pts_ = np.load(np_path + tail_paths[i])
        pts_ = pts_[slice_]
        for j, pt in enumerate(pts_):
            df[('Aditya', f'tail{j}')].loc['labeled-data', folder_name, img_names[i]] = pt
    df.to_hdf(f'/home/iyer_la/Documents/tadpole_stage57_Cam2-Aditya-2022-11-04/labeled-data/{folder_name}/CollectedData_Aditya.h5', 'df_with_missing')
    df.to_csv(f'/home/iyer_la/Documents/tadpole_stage57_Cam2-Aditya-2022-11-04/labeled-data/{folder_name}/CollectedData_Aditya.csv')
    return df

def get_boundary(x, b = 2048, margin = 300):
    x_mean = np.nanmean(x[:6])
    if x_mean < margin/2:
        return 0, margin
    if x_mean > b - margin/2:
        return b-margin, b
    return int(x_mean - margin/2), int(x_mean + margin/2)

def crop_folder(path_):
    df_ = pd.read_hdf(f'{path_}CollectedData_Aditya.h5')
    cols_ = df_.index
    for c in cols_[1:]:
        A = np.array(df_.loc[c])
        x = A[::2]
        y = A[1::2]
        x_range = get_boundary(x, 2048, 800)
        y_range = get_boundary(y, 1536, 1200)
        I = cv2.imread(path_ + c[2])
        A[::2] = x - x_range[0]
        A[1::2] = y - y_range[0]
        df_.loc[c] = A
        I = I[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        cv2.imwrite(path_ + c[2], I)
    df_.to_hdf(f'{path_}CollectedData_Aditya.h5', 'df_with_missing')
    df_.to_csv(f'{path_}CollectedData_Aditya.csv')

folders = os.listdir(f'/home/iyer_la/Documents/tadpole_stage57_Cam2-Aditya-2022-11-04/labeled-data/')
for folder_name in folders:
    df = add_tail(folder_name)
    crop_folder('/home/iyer_la/Documents/tadpole_stage57_Cam2-Aditya-2022-11-04/labeled-data/' + folder_name + '/')
label = True
move_to_folder = True

if label == True:
    import deeplabcut
    deeplabcut.check_labels('/home/iyer_la/Documents/tadpole_stage57_Cam2-Aditya-2022-11-04/config.yaml', draw_skeleton=True)

if move_to_folder == True:
    os.makedirs(f'/home/iyer_la/Documents/tadpole_stage57_Cam2-Aditya-2022-11-04/ALL/', exist_ok = True)
    import shutil
    p = '/home/iyer_la/Documents/tadpole_stage57_Cam2-Aditya-2022-11-04/labeled-data/'
    p1 = [i for i in os.listdir(p) if 'labeled' in i]
    count = 0
    for i in p1:
        files_ = [p + i + '/' + j for j in os.listdir(p + i)]
        for f in files_:
            shutil.copy(f, f'/home/iyer_la/Documents/tadpole_stage57_Cam2-Aditya-2022-11-04/ALL/{count}.png')
            count+=1

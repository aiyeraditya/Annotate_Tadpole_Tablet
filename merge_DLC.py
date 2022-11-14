'''
Aditya Iyer
Merge DeepLabCut datasets
'''

# import deeplabcut
import os, sys, shutil
import numpy as np
import pandas as pd


# Step1 : Create a new DeepLabCut project
# Step2 : Edit the config files to add videos in it. I did this manually
# Step3 : Move videos
# Step4 : Move the label folders
# Step5 : Change the Hdf files to reflect the change
# Step6 : Check labels with deeplabcut.check_labels
# Step7 : Move folder to soma
# Step8 : Run deeplabcut.create_training_dataset from soma
# Step9 : Submit job to SLURM

def make_new_project():
    deeplabcut.create_new_project('tadpole_stage57', 'Aditya', [], multianimal = False)

def move_videos():
    for cam_num in range(3):
        p1 = f'/home/iyer_la/Documents/tadpole_stage57_Cam{cam_num}-Aditya-2022-11-04/videos/'
        p2 = '/home/iyer_la/Documents/tadpole_stage57-Aditya-2022-11-14/videos/'
        for i in os.listdir(p1):
            shutil.copy(p1 + i, p2 + f'C{cam_num}_' + i)

def move_label_folders():
    for cam_num in range(3):
        p1 = f'/home/iyer_la/Documents/tadpole_stage57_Cam{cam_num}-Aditya-2022-11-04/labeled-data/'
        p2 = '/home/iyer_la/Documents/tadpole_stage57-Aditya-2022-11-14/labeled-data/'
        for i in os.listdir(p1):
            if 'labeled' in i:
                continue
            shutil.copytree(p1 + i, p2 + f'C{cam_num}_' + i.split('.')[0])

def change_hdf_files():
    path_ = '/home/iyer_la/Documents/tadpole_stage57-Aditya-2022-11-14/labeled-data/'
    for folder_ in os.listdir(path_):
        df = pd.read_hdf(f'{path_}{folder_}/CollectedData_Aditya.h5')
        df.index = df.index.set_levels([folder_], level = 1)
        df.to_hdf(f'{path_}{folder_}/CollectedData_Aditya.h5', 'df_with_missing')
        df.to_csv(f'{path_}{folder_}/CollectedData_Aditya.csv')

change_hdf_files()

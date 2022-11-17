import matplotlib.pyplot as plt
import aniposelib
import numpy as np
import pandas as pd
import os, sys, shutil, cv2
from PIL import Image
#import matplotlib


def get_traces(img_n, folder):
    p0 = np.load(f'/home/iyer_la/Documents/TailAnnotated/tadpole_stage57_Cam0-Aditya-2022-11-04/labeled-data/{folder}/Camera1img{img_n}_pts00000.npy')
    p1 = np.load(f'/home/iyer_la/Documents/TailAnnotated/tadpole_stage57_Cam1-Aditya-2022-11-04/labeled-data/{folder}/Camera1img{img_n}_pts00000.npy')
    p2 = np.load(f'/home/iyer_la/Documents/TailAnnotated/tadpole_stage57_Cam2-Aditya-2022-11-04/labeled-data/{folder}/Camera1img{img_n}_pts00000.npy')
    return p0, p1, p2

def get_imgs(img_n, folder):
    with Image.open(f'/home/iyer_la/Documents/TailAnnotated/tadpole_stage57_Cam0-Aditya-2022-11-04/labeled-data/{folder}/img{img_n}.png') as im:
        p0 = np.array(im)
    with Image.open(f'/home/iyer_la/Documents/TailAnnotated/tadpole_stage57_Cam1-Aditya-2022-11-04/labeled-data/{folder}/img{img_n}.png') as im:
        p1 = np.array(im)
    with Image.open(f'/home/iyer_la/Documents/TailAnnotated/tadpole_stage57_Cam2-Aditya-2022-11-04/labeled-data/{folder}/img{img_n}.png') as im:
        p2 = np.array(im)
    return p0, p1, p2

def get_neighbours(shp, curr_):
    x = range(max(0, curr_[0] - 1),min(curr_[0] + 2, shp[0]))
    y = range(max(0, curr_[1] - 1),min(curr_[1] + 2, shp[1]))
    neighbours = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)
    return neighbours

def djikstra_walk(array_, start_, end_):
    '''
    Finds the shortest path through the array_ using Djikstra's algorithm
    '''
    #Initiate all the required arrays
    #curr_ = np.array([0, np.argmin(residuals_[0])])
    curr_ = np.array(start_, dtype = np.int32)
    visited_ = np.zeros(array_.shape, dtype = np.bool_)
    previous_ = np.zeros((array_.shape[0], array_.shape[1], 2), dtype = np.int32)
    previous_[tuple(curr_)] = [-1, -1]
    distances_ = np.ones(array_.shape)*np.inf
    distances_[tuple(curr_)] = array_[tuple(curr_)]
    path_ = []

    while True:
        visited_[tuple(curr_)] = 1
        min_ = np.inf
        neighbours = get_neighbours(array_.shape, curr_)
        for ng in neighbours:
            # Update Distances
            if distances_[tuple(curr_)] + array_[tuple(ng)] < distances_[tuple(ng)]:
                distances_[tuple(ng)] = distances_[tuple(curr_)] + array_[tuple(ng)]
                previous_[tuple(ng)] = curr_
        distances_1 = np.copy(distances_);
        distances_1[np.where(visited_)] = np.inf
        curr_ = np.array(np.unravel_index(np.argmin(distances_1), distances_1.shape))
        if curr_[0] == end_[0] and curr_[1] == end_[1]:
            break
    # Traceback the path from the linked list
    while (curr_ != [-1, -1]).all():
        path_.append(curr_)
        curr_ = previous_[tuple(curr_)]
    path_ = np.flip(path_, axis = 0)

    return path_

def plot_trace(i0, i1, i2, p0, p1, p2):
    fig, ax = plt.subplots(1,3,figsize = (16,4))
    ax[0].imshow(i0)
    ax[0].plot(p0[:,0], p0[:,1])

    ax[1].imshow(i1)
    ax[1].plot(p1[:,0], p1[:,1])

    ax[2].imshow(i2)
    ax[2].plot(p2[:,0], p2[:,1])
    return fig, ax

def calclulate_residuals(calibration_, p1, p2):
    pts_pairs = np.zeros((3, 100*100, 2))
    pts_pairs[0] = np.nan
    for i in range(100):
        for j in range(100):
            pts_pairs[1][i*100 + j] = p1[i]
            pts_pairs[2][i*100 + j] = p2[j]

    pts3d = calibration_.triangulate(pts_pairs, progress=True)
    reproj_err = calibration_.reprojection_error(pts3d, pts_pairs, mean=True)
    residuals_ = np.reshape(reproj_err, (100, 100))
    return residuals_

def get_endpts(residuals_):

    if(min(residuals_[:,0]) < min(residuals_[0])):
        start_ = [np.argmin(residuals_[:,0]), 0]
    else:
        start_ = [0, np.argmin(residuals_[0])]

    if(min(residuals_[:,99]) < min(residuals_[99])):
        end_ = [np.argmin(residuals_[:,99]), 99]
    else:
        end_ = [99, np.argmin(residuals_[99])]

    # fig, ax = plt.subplots()
    # ax.matshow(residuals_, cmap = plt.cm.Blues, vmin =0, vmax = 50)
    # plt.plot(start_[1], start_[0], 'r*')
    # plt.plot(end_[1], end_[0], 'r*')

    return start_, end_

def plot_path(residuals_, path_):
    fig, ax = plt.subplots(figsize = (12,12))
    ax.matshow(residuals_)
    ax.plot(path_[:,1], path_[:,0], '.')
    return fig, ax

def reproject(calibration_, path_, p1, p2):

    slice_ = np.rint(np.linspace(0, len(path_)-1, 10)).astype(np.int32)
    x = path_[slice_,0]
    y = path_[slice_,1]
    pts_ = np.array([p1[x], p1[x], p2[y]])
    pts_[0] = np.nan
    tail3d = calibration_.triangulate(pts_)
    calibration_.reprojection_error(tail3d, pts_, mean = True)
    pts_reproj = calibration_.project(tail3d)
    return tail3d, pts_reproj, pts_

def main(folder, img_n):
    p0, p1, p2 = get_traces(img_n, folder)
    i0, i1, i2 = get_imgs(img_n, folder)
    pts_ = np.array([p0, p1, p2]).reshape(3, -1, 2)

    residuals_ = calclulate_residuals(calibration_, p1, p2)
    start_, end_ = get_endpts(residuals_)

    path_ = djikstra_walk(residuals_, start_, end_)
    #fig2, ax2 = plot_path(residuals_, path_)

    tail3d, pts_reproj, pts_subset = reproject(calibration_, path_, p1, p2);

    # Gotta save the images now
    for p in pts_reproj[0]:
        i0 = cv2.circle(i0, p.astype(np.int32), radius=4, color=(0, 255, 255), thickness=-1)
    for p in pts_reproj[1]:
        i1 = cv2.circle(i1, p.astype(np.int32), radius=4, color=(0, 255, 255), thickness=-1)
    for p in pts_reproj[2]:
        i2 = cv2.circle(i2, p.astype(np.int32), radius=4, color=(0, 255, 255), thickness=-1)
    cv2.imwrite(f'/home/iyer_la/Documents/TailAnnotated/DANNCE/img_C0_{folder}_{img_n}.png', i0)
    cv2.imwrite(f'/home/iyer_la/Documents/TailAnnotated/DANNCE/img_C1_{folder}_{img_n}.png', i1)
    cv2.imwrite(f'/home/iyer_la/Documents/TailAnnotated/DANNCE/img_C2_{folder}_{img_n}.png', i2)
    np.save(f'/home/iyer_la/Documents/TailAnnotated/DANNCE/points_{folder}_{img_n}.npy', tail3d)
    np.save(f'/home/iyer_la/Documents/TailAnnotated/DANNCE/reprojerr_{folder}_{img_n}.npy', calibration_.reprojection_error(tail3d, pts_subset, mean = True))



if __name__ == '__main__':
    calibration_ = aniposelib.cameras.CameraGroup.load('/mnt/soma_cifs/Iyer/Filming/02112022a/calibrationwater_1515/calibration.toml')
    #img_n = '0593'
    #folder = '76_9605-11135'
    for folder in os.listdir('/home/iyer_la/Documents/TailAnnotated/tadpole_stage57_Cam0-Aditya-2022-11-04/labeled-data/'):
        imgs_ = [i[3:7] for i in os.listdir(f'/home/iyer_la/Documents/TailAnnotated/tadpole_stage57_Cam0-Aditya-2022-11-04/labeled-data/{folder}/') if '.png' in i]
        for im in imgs_:
            try:
                main(folder, im)
            except:
                print(f"Error at {folder} {im}")

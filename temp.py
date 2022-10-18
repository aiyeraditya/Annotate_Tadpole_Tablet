#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:26:15 2022

@author: idyer_la
"""

import matplotlib.pyplot as plt
import cv2
from stereo_reproject import get_epipolar_line, get_calibration
import numpy as np
from scipy.spatial.transform import Rotation as R
import utils
from bokeh.plotting import figure, show
from scipy.interpolate import interp1d



def plot_line(coeffs, xlim):
    '''
    Given the coefficients a, b, c of the ax + by + c = 0, 
    plot the line within the given x limits.
    ax + by + c = 0 => y = (-ax - c) / b
    '''
    a, b, c = coeffs
    x = np.linspace(xlim[0], xlim[1], 100)
    y = (a * x + c) / -b
    return x, y


cali = '/mnt/soma_cifs/Iyer/Filming/16092022a/calibrationwater_1130/calibration.toml'
img1 = cv2.imread('/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/Camera0/frame192.jpg', 0);
img2 = cv2.imread('/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/Camera1/frame192.jpg', 0);
img3 = cv2.imread('/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/Camera2/frame182.jpg', 0);

# plt.imshow(img1)
# plt.plot(1380, 1080, '*r')

# X = np.random.rand(100)*2048
# Y = np.random.rand(100)*1536

# fig, ax = plt.subplots()
# plt.imshow(img2)
# plt.xlim([0, 2048])
# plt.ylim([1536, 0])
# for i in range(100):
#     l = get_epipolar_line(np.array([X[i], Y[i]]), 2, 1, cali)

#     A = plot_line(l, [0, 2048])
#     plt.plot(A[0], A[1])

l = get_epipolar_line(np.array([1373., 1124.]), 0,1, cali)
fig, ax = plt.subplots()
A = plot_line(l, [0, 2048])
plt.plot(A[0], A[1])
plt.imshow(img2)

#%%

import cv2
v1 = cv2.VideoCapture('/mnt/soma_cifs/Iyer/Filming/16092022a/Camera0/25.mp4');
v1.set(1, 3200)
for i in range(400):
    _, frame = v1.read()
    cv2.imwrite(f'/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/Camera0/frame{i}.jpg', frame)

v1.release()

#%%

img1 = cv2.imread('/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/Camera0/frame0.jpg', 0);
plt.plot(1380, 1080, '.r')
plt.imshow(img1)

#%%

def get_xyz(camera1_coords, cam1, 
            camera2_coords, cam2,
            camera3_coords = None, cam3 = None):

    camera1_P = cam1['matrix'] @ get_extrinsic_matrix(cam1)[:3, :]
    camera2_P = cam2['matrix'] @ get_extrinsic_matrix(cam2)[:3, :]
    if cam3 is not None:
        camera3_P = cam3['matrix'] @ get_extrinsic_matrix(cam3)[:3, :]

    camera1_u, camera1_v = camera1_coords
    camera1_vect1 = camera1_v*camera1_P[2,:]-camera1_P[1,:]
    camera1_vect2 = camera1_P[0,:] - camera1_u*camera1_P[2,:]
    
    camera2_u, camera2_v = camera2_coords
    camera2_vect1 = camera2_v*camera2_P[2,:]-camera2_P[1,:]
    camera2_vect2 = camera2_P[0,:] - camera2_u*camera2_P[2,:]
    
    full_matrix = np.row_stack((camera1_vect1, camera1_vect2, camera2_vect1, camera2_vect2))
    A = full_matrix[:, :3]
    b = full_matrix[:, 3].reshape((4, 1))
    soln = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(-b)
    return soln

def get_extrinsic_matrix(cam):
    rot = cam['rotation']
    rot = R.from_rotvec(rot).as_matrix()
    translation = cam['translation']
    extrinsic_matrix = np.identity(4)
    extrinsic_matrix[:3, :3] = rot
    extrinsic_matrix[:3, 3] = translation
    #extrinsic_matrix = extrinsic_matrix[:3, :]
    return extrinsic_matrix

def reproject(xyz, cam):
    cam_P = cam['matrix'] @ get_extrinsic_matrix(cam)[:3,:]
    xyz = np.array([xyz[0], xyz[1], xyz[2], 1])
    xy = cam_P @ xyz
    xy = xy/xy[2]
    return xy[:2]

c0, c1, c2 = get_calibration(cali)
# xyz = get_xyz([1380, 1080], c0, [1420, 1200], c1)
# xyz_h = np.append(xyz.T[0], 1).T
# xy0 = reproject(xyz_h, c0)
# xy1 = reproject(xyz_h, c1)

#%%


from aniposelib.cameras import Camera, CameraGroup


def get_reprojection_cube(points_xy, cali):
    calib_anipose = CameraGroup.load(cali)
    xyz = calib_anipose.triangulate(points_xy, progress = True)
    err_ = calib_anipose.reprojection_error(xyz, points_xy, mean=True)
    print(err_)
    err_mean = np.reshape(err_, (100, 100, 100));
    return err_mean

uv0 = np.load('Camera1_Img0000_pts0000.npy')
uv1 = np.load('Camera2_Img0000_pts0000.npy')
uv2 = np.load('Camera3_Img0000_pts0000.npy')

img0 = cv2.imread('/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/8/Camera0/frame47.jpg', 0);
img1 = cv2.imread('/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/8/Camera1/frame30.jpg', 0);
img2 = cv2.imread('/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/8/Camera2/frame47.jpg', 0);

cali = '/mnt/soma_cifs/Iyer/Filming/16092022a/calibrationwater_1130/calibration.toml'


xyz = np.zeros((100,100,100), dtype = np.float64);

points_xy = np.zeros((3, 100*100*100, 2));
count = 0
for p0 in uv0:
    for p1 in uv1:
        for p2 in uv2:
            points_xy[0][count] = p0
            points_xy[1][count] = p1
            points_xy[2][count] = p2
            count+= 1
err_mean = get_reprojection_cube(points_xy, cali)
            

#%%
def get_pos(N, k):
    i = N // 100
    j = N % 100
    return np.array([i, j, k], dtype = np.int16)

pos_ = [get_pos(np.argmin(err), i) for i, err in enumerate(err_mean)]

pts = np.array([[uv0[k], uv1[i], uv2[j]] for i,j,k in pos_])

# THis is not the best way to make use of the data we are generating

fig, ax = plt.subplots()
plt.imshow(img2, cmap=plt.get_cmap('gray'))
plt.plot(pts[::14,2,0], pts[::14, 2, 1], '.')
plt.plot(uv2[:,0], uv2[:,1], alpha = 0.5)
# plt.xlim([300, 800])
# plt.ylim([1100, 800])


fig, ax = plt.subplots()
plt.imshow(img1, cmap=plt.get_cmap('gray'))
plt.plot(pts[::14,1,0], pts[::14, 1, 1], '.')
plt.plot(uv1[:,0], uv1[:,1], alpha = 0.5)
# plt.xlim([1300, 1950])
# plt.ylim([1300, 800])


fig, ax = plt.subplots()
plt.imshow(img0, cmap=plt.get_cmap('gray'))
plt.plot(pts[::14,0,0], pts[::14, 0, 1], '.')
plt.plot(uv0[:,0], uv0[:,1], alpha = 0.5)
# plt.xlim([1300, 1800])
# plt.ylim([1536, 1000])


#%%

xyz = np.zeros((100,100), dtype = np.float64);

points_xy = np.zeros((3, 100*100, 2));
count = 0
for p0 in uv1:
    for p1 in uv2:
        points_xy[0][count] = [np.nan, np.nan]
        points_xy[1][count] = p0
        points_xy[2][count] = p1
        count+= 1
err_mean = get_reprojection_cube(points_xy, cali)
print(np.min(err_mean))
plt.imshow(err_mean)

#%%

plt.imshow(img0)
plt.plot(1420.26770966, 1101.51923004, '.')

fig, ax = plt.subplots()

plt.imshow(img1)
plt.plot(1493.67133977, 1185.07069392, '.')


#%%

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_GRAY2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img



fig, ax = plt.subplots(figsize = (6, 6.5))
plt.imshow(img2 + 20, cmap=plt.get_cmap('gray'))
#plt.plot(pts[::14,2,0], pts[::14, 2, 1], '.')
plt.plot(uv2[:,0], uv2[:,1], alpha = 1)
plt.xlim([225, 875])
plt.ylim([1250, 650])
plt.axis('off')
plt.savefig('Cam2_TailAnnotated.png')


fig, ax = plt.subplots()
plt.imshow(img1 + 25, cmap=plt.get_cmap('gray'))
#plt.plot(pts[::14,1,0], pts[::14, 1, 1], '.')
plt.plot(uv1[:,0], uv1[:,1], alpha = 1)
plt.xlim([1300, 1950])
plt.ylim([1350, 750])
plt.axis('off')
plt.savefig('Cam1_TailAnnotated.png')

fig, ax = plt.subplots()
plt.imshow(img0, cmap=plt.get_cmap('gray'))
#plt.plot(pts[::14,0,0], pts[::14, 0, 1], '.')
plt.plot(uv0[:,0], uv0[:,1], alpha = 1)
plt.xlim([1225, 1875])
plt.ylim([1536, 936])
plt.axis('off')
plt.savefig('Cam0_TailAnnotated.png')



#%%


im0 = cv2.imread('/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/Camera0/frame110.jpg')
im1 = cv2.imread('/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/Camera1/frame110.jpg')
im2 = cv2.imread('/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/Camera2/frame100.jpg')


path1 = "Camera1_Img0000_pts0000.npy"
path2 = "Camera2_Img0000_pts0000.npy"
path3 = "Camera3_Img0000_pts0000.npy"
p1 = np.load(path1)
p2 = np.load(path2)
p3 = np.load(path3)
plt.plot(p1[:,0], p1[:,1])
plt.plot(p2[:,0], p2[:,1])
plt.plot(p3[:,0], p3[:,1])
plt.xlim([0, 2048])
plt.ylim([0, 1536])

ani = CameraGroup.load(cali)

def spaced_ar(cx, N):
    ''' Returns N equidistant points along the curve cx'''
    dist_sum = np.zeros(xyz_corr.shape[0], dtype = np.float64)
    dist_sum[1:] = np.cumsum(np.linalg.norm(np.diff(xyz_corr, axis = 0), axis = 1))
    spl0 = interp1d(dist_sum, cx[:,0])
    spl1 = interp1d(dist_sum, cx[:,1])
    spl2 = interp1d(dist_sum, cx[:,2])
    spaced_l = np.linspace(0, dist_sum[-1], N+1)
    return np.array([spl0(spaced_l), spl1(spaced_l), spl2(spaced_l)])

pts2d = np.zeros((3,100*100,2), dtype = np.float64)
pts2d[2] = np.nan
for i in range(100):
    for j in range(100):
            pts2d[0][100*i + j] = p1[i]
            pts2d[1][100*i + j] = p2[j]

triangulated = ani.triangulate(pts2d, progress = True)
err_ = ani.reprojection_error(triangulated, pts2d, mean = True)
err_2d = err_.reshape((100, 100))

path_ = utils.get_path(err_2d)

xyz_corr = np.array([triangulated[i*100 + j] for i,j in path_])
xyz_corr_spaced = spaced_ar(xyz_corr, 20)

uvs = ani.project(xyz_corr_spaced.T)
fig,ax = plt.subplots(1,3, figsize = (30,10))
ax[2].plot(uvs[2,:,0], uvs[2,:,1], '.-', alpha = 0.6)
ax[2].plot(p3[:,0], p3[:,1], alpha = 0.6)
ax[2].imshow(im2)
# ax[2].set_xlim([1500, 2000])
# ax[2].set_ylim([1000, 400])

ax[1].plot(uvs[1,:,0], uvs[1,:,1], '.-', alpha = 0.6)
ax[1].plot(p2[:,0], p2[:,1], alpha = 0.6)
ax[1].imshow(im1)
# ax[1].set_xlim([1500, 2000])
# ax[1].set_ylim([1200, 200])

ax[0].plot(uvs[0,:,0], uvs[0,:,1], '.-', alpha = 0.6)
ax[0].plot(p1[:,0], p1[:,1], alpha = 0.6)
ax[0].imshow(im0)
# ax[0].set_xlim([100, 400])
# ax[0].set_ylim([1400, 1100])

#%%

plt.imshow(err_2d)
#plt.plot(good_pts[:,1], good_pts[:,0], '*')
plt.plot(path_[:,1], path_[:,0])


#%% Djikstra for shortest path


def handle_edge(nd):
    
    def check_val(i):
        if i < 0:
            return 0
        if i > 99:
            return 99
        return i
    nd[0] = check_val(nd[0])
    nd[1] = check_val(nd[1])
    return nd

def get_min_info(array):
    arg_ = np.argmin(array)
    min_ = array[arg_]
    return min_, arg_

def get_array_holes(err_2d, N):
    min_h, arg_h = get_min_info(err_2d[N])
    min_v, arg_v = get_min_info(err_2d[:,N])
    if min_h < min_v:
        hole_ = [N, arg_h]
    else:
        hole_ = [arg_v, N]
    return np.array(hole_)


def get_path(err_2d):

    jumps = np.array([[-1,-1], [-1,0], [-1, 1], [0, -1], [0,1], [1,-1], [1,0], [1,1]]);
    
    start_ = get_array_holes(err_2d, 0)
    end_ = get_array_holes(err_2d, 99)
    
    curr_ = np.array(start_);
    x0, y0 = curr_;
    cost_2d = np.ones((100,100))*1000;
    cost_2d[x0, y0] = err_2d[x0, y0]
    
    explored = np.zeros((100, 100))
    explored[x0, y0] = 1;
    new_ = np.copy(start_);
    prev = np.ones((100, 100, 2), dtype = np.uint8)*-1
    
    
    while not (curr_ == end_).all():
        min_node = 1001;
        cost_curr = cost_2d[curr_[0], curr_[1]];
        for jmp in jumps:
            x_, y_ = handle_edge(curr_ + jmp)
            if explored[x_, y_] == 1:   # We don't care about explored nodes
                continue
            
            if (cost_curr + err_2d[x_][y_]) < cost_2d[x_][y_]:
                cost_2d[x_][y_] = cost_curr + err_2d[x_][y_]
                prev[x_, y_] = curr_
            
            if cost_2d[x_][y_] < min_node:
                min_node = cost_2d[x_][y_]
                new_ = [x_, y_]
        curr_ = np.array(new_)
        explored[curr_[0], curr_[1]] = 1 #Mark as explored
    
    # Find the path from the prev linked list
    path_ = [end_]
    curr_ = path_[-1]
    while True:
        curr_ = path_[-1]
        if (curr_ == start_).all():
            break
        path_.append(prev[curr_[0], curr_[1]])
    path_ = np.flip(path_, axis =0)
    return path_
































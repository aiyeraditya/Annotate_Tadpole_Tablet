#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:39:06 2022

@author: Aditya Iyer
"""

import numpy as np
import os, cv2, ast
from scipy.spatial.transform import Rotation as R

def parse_calibration(str_cam):
    '''
    
    Function returns a dictionary containing the size, matrix, distortions, 
    rotation and translation for the camera based on the string str_cam

    Parameters
    ----------
    str_cam : string
        From the anipose calibration output file.

    Returns
    -------
    camera : Dictionary


    '''
    str_cam = str_cam.split('\n')
    camera = {}
    params = ['size', 'matrix', 'distortions', 'rotation', 'translation']
    for param in params:
        camera[param] = [np.array(ast.literal_eval(i.split('=')[1].strip())) 
                         for i in str_cam if param in i][0]
    return camera


def get_calibration(calib_file):
    '''
    Parameters
    ----------
    calib_file : File Path to Anipose Calibration TOML file

    Returns
    -------
    None
    
    Function Description
    -------
    '''
    with open(calib_file, 'r') as f:
        l = f.read()
        l = l.split('cam')

    cam0 = parse_calibration(l[1])
    cam1 = parse_calibration(l[2])
    cam2 = parse_calibration(l[3])
    return cam0, cam1, cam2

def get_rot_trans(cam):
    
    rot = cam['rotation']
    translation = cam['translation']
    return rot, translation

def get_fundamental_matrix(camA, camB):
    R_A, T_A = get_rot_trans(camA)
    R_B, T_B = get_rot_trans(camB)
    
    rotA = R.from_rotvec(R_A).as_matrix()
    rotB = R.from_rotvec(R_B).as_matrix()
    
    rotB = np.linalg.inv(rotA) @ rotB
    
    T_B = np.linalg.inv(rotA) @ (T_B - T_A)
    
    Tx = np.cross(T_B, np.identity(T_B.shape[0]) * -1)

    K1 = np.linalg.inv(camB['matrix']).T
    K2 = np.linalg.inv(camA['matrix'])
    fundamental_matrix = K1 @ Tx @ rotB @ K2
    return fundamental_matrix


def get_epipolar_line(pt, cam_ID1, cam_ID2, calib_file):

    C = get_calibration(calib_file)
    F = get_fundamental_matrix(C[cam_ID1], C[cam_ID2])
    pt_3 = np.ones((3,))
    pt_3[:2] = pt
    coeffs = (F @ pt_3)
    print(coeffs)
    return coeffs











# # extrinsic_matrix = np.identity(4)
# # extrinsic_matrix[:3, :3] = rotB
# # extrinsic_matrix[:3, 3] = T_B
# # extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
# # projection_matrix = camB['matrix'] @ extrinsic_matrix


# def get_extrinsic_matrix(cam):
    
#     rot = cam['rotation']
#     rot = R.from_rotvec(rot).as_matrix()
#     translation = cam['translation']
#     extrinsic_matrix = np.identity(4)
#     extrinsic_matrix[:3, :3] = rot
#     extrinsic_matrix[:3, 3] = translation
#     extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
#     return extrinsic_matrix

# def get_essential_matrix(camA, camB):
    
#     extrinsicA = get_extrinsic_matrix(camA)
#     extrinsicB = get_extrinsic_matrix(camB)
#     extrinsicAB = extrinsicB @ np.linalg.inv(extrinsicA)
#     rotationAB = extrinsicAB[:3, :3]
#     t_AB = extrinsicAB[:3, 3]
#     translation_cross = np.cross(t_AB, np.identity(t_AB.shape[0]) * -1)
#     essential_matrix = translation_cross @ rotationAB
#     return essential_matrix

# def get_fundamental_matrix(camA, camB):
#     essential_matrix = get_essential_matrix(camA, camB);
#     KB = camB['matrix']
#     KA = camA['matrix']
#     fundamental_matrix = np.linalg.inv(KA).T @ essential_matrix @ np.linalg.inv(KB)
#     return fundamental_matrix


# def get_epipolar_line(pt, cam_ID1, cam_ID2, calib_file):

#     C = get_calibration(calib_file)
#     F = get_fundamental_matrix(C[cam_ID1], C[cam_ID2])
#     pt_3 = np.ones((3,))
#     pt_3[:2] = pt
#     coeffs = (F @ pt_3).reshape(-1)
#     return coeffs


if __name__ == '__main__':
    print('You Should not be running this directly')
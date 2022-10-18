#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:26:22 2022

@author: idyer_la
"""

import numpy as np

from bokeh.plotting import figure, show
from bokeh.models import CustomJS, Slider, ColumnDataSource
from bokeh.layouts import row, column

def visualize_3Darray(A):
    source = ColumnDataSource(data=dict(image=[A[0, :, :]]))
    x0, y0 = source.data['image'][0].shape
    p = figure(tooltips=[("x", "$x{0.}"), ("y", "$y{0.}"), ("value", "@image")])
    p.image(image='image', x=0, y=0, dw=x0, dh=y0, source=source, palette="Viridis256")
    slider = Slider(start=0, end=(len(A) - 1), value=0, step=1, title="SliceNumber")
    code= """
        source.data = {
        image : [array[n.value]]
        }
    """
    callback = CustomJS(args=dict(source=source, n=slider, array = A), code = code)
    slider.js_on_change('value', callback);
    layout = column(slider, p)
    return layout

def visualize_image(img_):
    source = ColumnDataSource(data=dict(image=[img_]))
    x0, y0 = source.data['image'][0].shape
    p = figure(tooltips=[("x", "$x{0.}"), ("y", "$y{0.}"), ("value", "@image")])
    p.image(image='image', x=0, y=0, dw=x0, dh=y0, source=source, palette="Viridis256")
    return p



# Djikstra Algorithm

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


# Video Timestamp Locha

def get_firsttime(a):
    with open(a, 'r') as f:
        l = f.readlines()
        l = [i for i in l if 'FirstFrameTime' in i][0]
    return float(l.split(',')[1][1:-2])

def get_time(a, N = 0):
    t0 = get_firsttime(a + '_metadata.csv')
    dt = np.load(a + '_frametimes.npy')
    return t0 + dt[1, N]

def get_deltas(path_, vid_num):
    a = get_time(f'{path_}Camera0/{vid_num}')
    b = get_time(f'{path_}Camera1/{vid_num}')
    c = get_time(f'{path_}Camera2/{vid_num}')
    max_ = np.max([a,b,c])
    delA = np.round((max_ - a)*100)
    delB = np.round((max_ - b)*100)
    delC = np.round((max_ - c)*100)
    return np.array([delA, delB, delC], dtype = np.uint64)
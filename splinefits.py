#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 23:25:03 2022

@author: idyer_la
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x = []
y = []
with open('filename.txt') as f:
    lines = f.readlines()
    for line in lines:
        [xs,ys] = line.split(',')
        xs = float(xs)
        ys = float(ys)
        x.append(xs)
        y.append(ys)
        
x = np.array(x)
y = np.array(y)
plt.plot(x,y)

t = np.zeros(x.shape)
t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
t = np.cumsum(t)
t /= t[-1]

fx = interp1d(t,x)
fy = interp1d(t,y)

fig, ax = plt.subplots()
nt1 = np.linspace(0, 1, 100)
x1 = fx(nt1)
y1 = fy(nt1)
plt.plot(x,y)
plt.plot(x1, y1, '.')
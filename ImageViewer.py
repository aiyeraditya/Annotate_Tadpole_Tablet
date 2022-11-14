#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:21:19 2022

@author: idyer_la
"""

import sys
import os
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QIcon, QPixmap, QTabletEvent, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ImageViewer():
    def __init__(self, image_name, position, cam_ID, image_number = '0', scale_factor = 3.3):
        self.pixmap = QPixmap(image_name)
        self.img_ = np.asarray(Image.open(image_name))
        self.img_height, self.img_width, _ = self.img_.shape
        self.camera_name = cam_ID

        self.scale_factor = scale_factor
        self.viewer_width = int(self.pixmap.size().width()/self.scale_factor)
        self.viewer_height = int(self.pixmap.size().height()/self.scale_factor)
        self.xoffset = position[0]
        self.yoffset = position[1]
        self.position = QtCore.QRect(
            self.xoffset, self.yoffset, self.viewer_width, self.viewer_height)
        self.zoom_ROI = QtCore.QRect(0, 0, self.img_width, self.img_height)
        self.zoomFactor = 1;

        self.transformed_pts = []
        self.count = 0
        self.image_number = image_number

        self.filenames = []

    def update_image(self, image_name, image_num):
        self.pixmap = QPixmap(image_name)
        self.image_number = image_num

    def draw(self, painter):
        self.pixmap_cropped = self.pixmap.copy(
            self.zoom_ROI).scaledToHeight(self.viewer_height)
        painter.drawPixmap(self.position, self.pixmap_cropped)

    def set_zoom(self, hover_start, hover_end, pen_x, pen_y):
        self.hover_end = QtCore.QPoint(int(pen_x), int(pen_y))
        topleft = QtCore.QPoint(hover_start.x() - self.xoffset,
                                hover_start.y() - self.yoffset)
        bottomright = QtCore.QPoint(hover_end.x() - self.xoffset,
                                    hover_end.y() - self.yoffset)
        rect = QtCore.QRect(topleft, bottomright)
        if rect.height() < 0 or rect.width() < 0:
            self.zoomFactor = 1;
            self.zoom_ROI = QtCore.QRect(0, 0, self.img_width,
                                         self.img_height);

    def check_bounds(self, X1, mx, img_w):
        if X1 < 0:
            X1 = 0
        if X1 > img_w - mx:
            X1 = img_w - mx;
        X2 = X1 + mx
        return X1, X2

    def transform_pts(self, pen_x, pen_y):
        Px = (pen_x - self.xoffset)*self.scale_factor
        Py = (pen_y - self.yoffset)*self.scale_factor
        a,b = self.zoom_ROI.getCoords()[0:2]
        w = self.img_width / self.zoom_ROI.width()
        h = self.img_height / self.zoom_ROI.height()
        X = (Px/w + a)
        Y = (Py/h + b)
        return X,Y

    def inverse_pts(self, x, y):
        w = self.img_width / self.zoom_ROI.width()
        h = self.img_height / self.zoom_ROI.height()
        a,b = self.zoom_ROI.getCoords()[0:2]
        X = w*(x - a)/self.scale_factor + self.xoffset
        Y = h*(y - b)/self.scale_factor + self.yoffset
        return QtCore.QPointF(X,Y)

    def keyboard_zoom(self, pen_x, pen_y, zoom_in = True):
        '''
        Zoom into the region, with 80% of the screen if zoom_in is True
        '''
        Px, Py = self.transform_pts(pen_x, pen_y)
        # First Check that penPoint is within the image ROI
        if Px > self.img_width or Py > self.img_height:
            return
        if Px < 0 or Py < 0:
            return
        # If the current ROI is WxH, the new ROI must be W/1.1*H/1.1
        if zoom_in:
            self.zoomFactor *= 1.1;
        else:
            self.zoomFactor *= 0.9;
            self.zoomFactor = max(1, self.zoomFactor)
        img_w = self.img_width/(self.zoomFactor);
        topleft_x = Px - img_w/2;
        topleft_x, bottomright_x = self.check_bounds(topleft_x, img_w,
                                                     self.img_width)

        img_h = self.img_height/(self.zoomFactor);
        topleft_y = Py - img_h/2;
        topleft_y, bottomright_y = self.check_bounds(topleft_y, img_h,
                                                     self.img_height)
        topleft = QtCore.QPoint(int(topleft_x), int(topleft_y))
        bottomright = QtCore.QPoint(int(bottomright_x), int(bottomright_y))
        self.zoom_ROI = QtCore.QRect(topleft, bottomright)
        return

    def load_pts(self, path_):
        self.transformed_pts = [QtCore.QPointF(*p) for p in np.load(path_)]


    def collect_path(self):
        self.obtain_equidistantpts()
        self.save_pts()
        self.transformed_pts = []

    def obtain_equidistantpts(self):
        self.xpts = np.array([pt.x() for pt in self.transformed_pts])
        self.ypts = np.array([pt.y() for pt in self.transformed_pts])
        x = self.xpts
        y = self.ypts
        t = np.zeros(x.shape)
        t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
        t = np.cumsum(t)
        t /= t[-1]
        fx = interp1d(t, x)
        fy = interp1d(t, y)
        nt1 = np.linspace(0, 1, 100)
        self.equidistant_pts = np.transpose([fx(nt1), fy(nt1)])

    def save_pts(self):
        ptname_ = str(self.count).zfill(5)
        #imgname_ = str(self.image_number).zfill(5)
        imgname_ = self.image_number[:-4]
        np.save(f'{self.camera_name}{imgname_}_pts{ptname_}.npy',
                self.equidistant_pts)
        self.filenames.append(f'{self.camera_name}{imgname_}_pts{ptname_}.npy')
        self.count += 1

    def visualize_fit(self):
        fig, ax = plt.subplots()
        ax.plot(self.xpts, self.ypts)
        ax.plot(self.equidistant_pts[:, 0], self.equidistant_pts[:, 1], '*')
        plt.show()

    def save_npys(self):
        NPY = []
        for i in self.filenames:
            pts = np.load(i)
            NPY.append(pts)
            #os.remove(i)
        np.save(f'{self.camera_name}_pts.npy', NPY)

if __name__ == '__main__':
    print('This')

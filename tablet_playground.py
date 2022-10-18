# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys, os
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QIcon, QPixmap, QTabletEvent, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


class TabletSampleWindow(QWidget):
    def __init__(self, parent=None):
        super(TabletSampleWindow, self).__init__(parent)
        self.pen_is_down = False
        self.pen_x = 0
        self.pen_y = 0
        self.hover = False
        self.drawing = False
        self.text = "Ready to Accept Data"
        self.move(-9, 1920)
        frame_rect = app.desktop().frameGeometry()
        width, height = frame_rect.width(), frame_rect.height()
        self.resize(width, height)
        self.setWindowTitle("Sample Tablet Event Handling")
        self.pixmap = QPixmap("HappyFish.jpg")
        self.h = self.pixmap.size().height()
        self.xoffset = 50;
        self.yoffset = 50;
        self.zoom_ROI = QtCore.QRect(0,0,self.pixmap.size().width(), self.pixmap.size().height())
        self.pts = [];
        self.hover_start = [];
        self.hover_end = [];
        self.count = 0;
        self.filenames = [];
        
    def tabletEvent(self, tabletEvent):
        self.pen_x = tabletEvent.hiResGlobalX()
        self.pen_y = tabletEvent.hiResGlobalY()
        self.pen_pressure = int(tabletEvent.pressure() * 100)
        if tabletEvent.type() == QTabletEvent.TabletPress:
            press_type = tabletEvent.button(); # 1: Stylus 2: Button2 4: Button1
            if press_type == 1 and not self.hover:
                self.drawing = True
                self.text = "Collecting Points"
            elif press_type == 4:
                if self.drawing:
                    self.drawing = False
                    self.pts = [];
                self.hover = True
                self.hover_start = QtCore.QPoint(int(self.pen_x), int(self.pen_y) - 1960);
                self.text = "ZOOM Activated"
        elif tabletEvent.type() == QTabletEvent.TabletRelease:
            if self.drawing:
                self.drawing = False
                self.obtain_equidistantpts()
                self.save_pts()
                self.pts = [];
                self.text = f"Points Collected {self.count}"
            elif self.hover:
                self.hover = False
                self.hover_end = QtCore.QPoint(int(self.pen_x), int(self.pen_y) - 1960);
                self.set_zoom()
                self.text = "Ready to Accept Data"
            
        if self.drawing:
            self.pts.append(QtCore.QPointF(self.pen_x, self.pen_y - 1960));
        elif self.hover:
            self.hover_end = QtCore.QPoint(int(self.pen_x), int(self.pen_y) - 1960);
        tabletEvent.accept()
        self.update()
    
    def set_zoom(self):
        topleft = QtCore.QPoint(self.hover_start.x() - self.xoffset,
                                self.hover_start.y() - self.yoffset)
        bottomright = QtCore.QPoint(self.hover_end.x() - self.xoffset,
                                self.hover_end.y() - self.yoffset)
        self.zoom_ROI = QtCore.QRect(topleft, bottomright);

    def paintEvent(self, event):
        text = self.text
        painter = QPainter(self)
        painter.drawText(self.rect(), QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft , text)
        
        self.pixmap_cropped = self.pixmap.copy(self.zoom_ROI).scaledToHeight(self.h);
        w = self.pixmap_cropped.size().width()
        h = self.pixmap_cropped.size().height()
        painter.drawPixmap(QtCore.QRect(self.xoffset, self.yoffset, w, h), self.pixmap_cropped)
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(QtCore.Qt.red)
        painter.setPen(pen)
        
        if self.hover:
            painter.drawRect(QtCore.QRect(self.hover_start, self.hover_end)) 
            
        for point in self.pts:
            painter.drawPoint(point)
            
    def obtain_equidistantpts(self):
        self.xpts = np.array([pt.x() for pt in self.pts]);
        self.ypts = np.array([pt.y() for pt in self.pts]);
        x = self.xpts;
        y = self.ypts;
        t = np.zeros(x.shape)
        t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
        t = np.cumsum(t)
        t /= t[-1]
        fx = interp1d(t,x)
        fy = interp1d(t,y)
        nt1 = np.linspace(0, 1, 100)
        self.equidistant_pts = np.transpose([fx(nt1), fy(nt1)])
    
    def save_pts(self):
        np.save(f'pts_{self.count}.npy', self.equidistant_pts)
        self.filenames.append(f'pts_{self.count}.npy')
        self.count+=1
        
    def visualize_fit(self):
        fig,ax = plt.subplots()
        ax.plot(self.xpts, self.ypts);
        ax.plot(self.equidistant_pts[:,0], self.equidistant_pts[:,1], '*')
        plt.show()
        
    def save_npys(self):
        NPY = []
        for i in self.filenames:
            pts = np.load(i)
            NPY.append(pts)
            os.remove(i);
        np.save('pts.npy', NPY)
            
    def closeEvent(self, *args, **kwargs):
        self.save_npys()

app = QApplication(sys.argv)
mainform = TabletSampleWindow()
mainform.show()
app.exec_()


''' Things to Implement

1. Show Image of Tadpole on the Device
2. On PenDown and PenUp event, draw
3. LowPass filter the pen data
4. Show the Start and End pixel reconstructions on the other screens with Stereo
5. Once input on all screens is present, find the best spline fit, and overlay
the spline fit with the drawn curve
6. Accept the labeling and proceed OR reject the labeling and proceed
7. Quit and Save all

1. Add Image and Zoom feature
2. Figure out the transformation from tablet pixel coordinates to image coordinates



'''


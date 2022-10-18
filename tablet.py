# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QIcon, QPixmap, QTabletEvent, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from ImageViewer import ImageViewer    #This is a self-defined Class file. See ImageViewer.py

import utils


class TabletSampleWindow(QWidget):
    def __init__(self, parent=None):
        super(TabletSampleWindow, self).__init__(parent)
        self.pen_is_down = False
        self.pen_x = 0
        self.pen_y = 0
        self.hover = False
        self.drawing = False
        self.text = "Ready to Accept Data"
        self.move(-9, 1920)  # Move the Window to the tablet desktop
        frame_rect = app.desktop().frameGeometry()
        width, height = frame_rect.width(), frame_rect.height()
        self.resize(width, height)
        self.setWindowTitle("Sample Tablet Event Handling")
        
        self.vid_num = 25;
        self.Video1 = f'/mnt/soma_cifs/Iyer/Filming/16092022a/Camera0/{self.vid_num}.mp4';
        self.Video2 = f'/mnt/soma_cifs/Iyer/Filming/16092022a/Camera1/{self.vid_num}.mp4';
        self.Video3 = f'/mnt/soma_cifs/Iyer/Filming/16092022a/Camera2/{self.vid_num}.mp4';
        self.initialize_vidcap()
        self.offsets = utils.get_deltas('/mnt/soma_cifs/Iyer/Filming/16092022a/', self.vid_num)
        
        
        self.image_number = 100
        self.Viewer1 = ImageViewer(f'/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/Camera0/frame{self.image_number+10}.jpg',
                                   [50, 50], 'Camera1')
        self.Viewer2 = ImageViewer(f'/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/Camera1/frame{self.image_number+10}.jpg',
                                   [1050, 50], 'Camera2')
        self.Viewer3 = ImageViewer(f'/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/Camera2/frame{self.image_number}.jpg',
                                   [50, 550],  'Camera3')

        self.hover_start = []
        self.hover_end = []

        self.current_viewer = None
        self.curr_view_txt = None
        
    def initialize_vidcap(self):
        self.cap1 = cv2.VideoCapture(self.Video1)
        self.cap2 = cv2.VideoCapture(self.Video2)
        self.cap3 = cv2.VideoCapture(self.Video3)
    
    def close_vidcap(self):
        self.cap1.release()
        self.cap2.release()
        self.cap3.release()
        
    def which_viewer(self):
        if self.Viewer1.position.contains(int(self.pen_x), int(self.pen_y)):
            self.current_viewer = self.Viewer1
            self.curr_view_txt = 'Image 01'
        elif self.Viewer2.position.contains(int(self.pen_x), int(self.pen_y)):
            self.current_viewer = self.Viewer2
            self.curr_view_txt = 'Image 02'
        elif self.Viewer3.position.contains(int(self.pen_x), int(self.pen_y)):
            self.current_viewer = self.Viewer3
            self.curr_view_txt = 'Image 03'
        else:
            self.current_viewer = None
            self.curr_view_txt = 'NoViewer Active'

    def tabletEvent(self, tabletEvent):
        self.pen_x = tabletEvent.hiResGlobalX()
        self.pen_y = tabletEvent.hiResGlobalY() - 1960
        self.pen_pressure = int(tabletEvent.pressure() * 100)
        
        self.which_viewer();
        if self.current_viewer is None:
            return
            
        if tabletEvent.type() == QTabletEvent.TabletPress:
            press_type = tabletEvent.button()  #1: Stylus 2: Button2 4: Button1
            if press_type == 1 and not self.hover:
                self.start_draw()

            elif press_type == 4:
                self.buttonpress()
        elif tabletEvent.type() == QTabletEvent.TabletRelease:
            if self.drawing:
                self.drawing = False
                self.current_viewer.collect_path()
            elif self.hover:
                self.hover = False
                self.current_viewer.set_zoom(
                    self.hover_start, self.hover_end, self.pen_x, self.pen_y)
                self.text = "Ready to Accept"
                
        if self.drawing:
            X, Y = self.current_viewer.transform_pts(self.pen_x, self.pen_y)
            self.current_viewer.transformed_pts.append(QtCore.QPointF(X, Y))
            self.current_viewer.pts.append(QtCore.QPointF(self.pen_x, 
                                                          self.pen_y))
        elif self.hover:
            self.hover_end = QtCore.QPoint(int(self.pen_x), int(self.pen_y))
            
        tabletEvent.accept()
        self.update()

    def start_draw(self):
        self.drawing = True
        self.text = f"Collecting Points for {self.curr_view_txt}";

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            self.closeEvent()
            
        if event.key() == QtCore.Qt.Key_Q:
            self.closeEvent()

        elif event.key() == QtCore.Qt.Key_Plus:
            self.current_viewer.keyboard_zoom(self.pen_x, self.pen_y, 
                                              zoom_in = True)
            self.update()

        elif event.key() == QtCore.Qt.Key_Minus:
            self.current_viewer.keyboard_zoom(self.pen_x, self.pen_y, 
                                              zoom_in = False)
            self.update()
            
        elif event.key() == QtCore.Qt.Key_N:
            self.image_number = min(self.image_number + 10, 650)
            self.Viewer1.update_image(f'/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/Camera0/frame{self.image_number}.jpg', self.image_number)
            self.Viewer2.update_image(f'/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/Camera1/frame{self.image_number}.jpg', self.image_number)
            self.Viewer3.update_image(f'/mnt/soma_cifs/Iyer/Filming/AnnotationTemp/Camera2/frame{self.image_number}.jpg', self.image_number)
            self.Viewer1.pts = []
            self.Viewer2.pts = []
            self.Viewer3.pts = []
            self.update()
            
        event.accept()

    def buttonpress(self):
        if self.drawing:
            self.drawing = False
            self.current_viewer.pts = []
        self.hover = True
        self.hover_start = QtCore.QPoint(int(self.pen_x), int(self.pen_y))
        self.text = f"ZOOM Activated for {self.curr_view_txt}";

    def paintEvent(self, event):
        text = self.text
        painter = QPainter(self)
        painter.drawText(self.rect(), QtCore.Qt.AlignTop |
                         QtCore.Qt.AlignLeft, text)
        self.Viewer1.draw(painter)
        self.Viewer2.draw(painter)
        self.Viewer3.draw(painter)

        pen = QPen()
        pen.setWidth(2)
        pen.setColor(QtCore.Qt.red)
        painter.setPen(pen)

        if self.hover:
            painter.drawRect(QtCore.QRect(self.hover_start, self.hover_end))
        for point in self.Viewer1.pts:
            painter.drawPoint(point)
        for point in self.Viewer2.pts:
            painter.drawPoint(point)
        for point in self.Viewer3.pts:
            painter.drawPoint(point)

    def closeEvent(self, *args, **kwargs):
        self.close_vidcap()
        self.Viewer1.save_npys()
        self.Viewer2.save_npys()
        self.Viewer3.save_npys()
        self.deleteLater()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainform = TabletSampleWindow()
    mainform.show()
    app.exec_()


''' Things to Implement

1. Show Multiple Images. Handle pixel offsets for each DONE
4. Show the Start and End pixel reconstructions on the other screens with Stereo
5. Once input on all screens is present, find the best spline fit, and overlay
the spline fit with the drawn curve
6. Accept the labeling and proceed OR reject the labeling and proceed
7. Quit and Save all

2. Figure out the transformation from tablet pixel coordinates to image coordinates


06.OCT.2022 Task Lists

1. 3D Pixel Reconstructions and Lines
2. Accept/ReDo points
3. Returning to points

'''
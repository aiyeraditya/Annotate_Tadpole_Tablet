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
import cv2, os

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

        self.image_number = 3200
        self.Viewer1 = ImageViewer(f'/mnt/soma_cifs/Iyer/RequestingHelpOCT2022/02112022a_43/Camera0/img{str(self.image_number).zfill(5)}.png',
                                   [250, 50], 'Camera1', self.image_number, 1.8)
        self.hover_start = []
        self.hover_end = []

        self.current_viewer = None
        self.curr_view_txt = None

    def which_viewer(self):
        if self.Viewer1.position.contains(int(self.pen_x), int(self.pen_y)):
            self.current_viewer = self.Viewer1
            self.curr_view_txt = 'Image 01'
        else:
            self.current_viewer = None
            self.curr_view_txt = 'NoViewer Active'

    def tabletEvent(self, tabletEvent):
        self.pen_x = tabletEvent.hiResGlobalX()
        self.pen_y = tabletEvent.hiResGlobalY() - 1240
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
            # self.current_viewer.pts.append(QtCore.QPointF(self.pen_x,
            #                                               self.pen_y))
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

        if event.key() == QtCore.Qt.Key_D:
            self.delete_pts()

        elif event.key() == QtCore.Qt.Key_Plus:
            self.current_viewer.keyboard_zoom(self.pen_x, self.pen_y,
                                              zoom_in = True)
            self.update()

        elif event.key() == QtCore.Qt.Key_Minus:
            self.current_viewer.keyboard_zoom(self.pen_x, self.pen_y,
                                              zoom_in = False)
            self.update()

        elif event.key() == QtCore.Qt.Key_N:
            self.image_number += 1
            self.Viewer1.count = 0
            print(f'Loaded /mnt/soma_cifs/Iyer/RequestingHelpOCT2022/02112022a_43/Camera0/img{str(self.image_number).zfill(5)}.png')
            self.Viewer1.update_image(f'/mnt/soma_cifs/Iyer/RequestingHelpOCT2022/02112022a_43/Camera0/img{str(self.image_number).zfill(5)}.png', self.image_number)
            self.update()

        elif event.key() == QtCore.Qt.Key_P:
            self.image_number -= 1
            self.Viewer1.count = 0
            print(f'Loaded /mnt/soma_cifs/Iyer/RequestingHelpOCT2022/02112022a_43/Camera0/img{str(self.image_number).zfill(5)}.png')
            self.Viewer1.update_image(f'/mnt/soma_cifs/Iyer/RequestingHelpOCT2022/02112022a_43/Camera0/img{str(self.image_number).zfill(5)}.png', self.image_number)
            if os.path.exists(f'Camera1_Img{str(self.image_number).zfill(5)}_pts00000.npy'):
                self.Viewer1.load_pts(f'Camera1_Img{str(self.image_number).zfill(5)}_pts00000.npy')
            self.update()

        event.accept()

    def delete_pts(self):
        npys = [i for i in os.listdir() if 'npy' in i and str(self.image_number) in i]
        for i in npys:
            os.remove(i)
            self.Viewer1.filenames.remove(i)
            print(f'Deleted {i}')
        self.Viewer1.transformed_pts = []
        self.Viewer1.count = 0;
        self.update()
        print(self.Viewer1.filenames)

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

        pen = QPen()
        pen.setWidth(2)
        pen.setColor(QtCore.Qt.red)
        painter.setPen(pen)

        if self.hover:
            painter.drawRect(QtCore.QRect(self.hover_start, self.hover_end))

        if os.path.exists(f'Camera1_Img{str(self.image_number).zfill(5)}_pts00000.npy'):
            pts = [QtCore.QPointF(*p) for p in np.load(f'Camera1_Img{str(self.image_number).zfill(5)}_pts00000.npy')]
            for point in pts:
                pt = self.Viewer1.inverse_pts(point.x(), point.y())
                painter.drawPoint(pt)
        if self.Viewer1.transformed_pts:
            for point in self.Viewer1.transformed_pts:
                pt = self.Viewer1.inverse_pts(point.x(), point.y())
                painter.drawPoint(pt)

    def closeEvent(self, *args, **kwargs):
        self.Viewer1.save_npys()
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

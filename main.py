import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from collections import defaultdict
import torch.nn.functional as F
import cv2 as cv
from numpy import *
from scipy.spatial import distance
import matplotlib.cm as cm
import PyQt5
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QLabel, QFrame, QGridLayout, QFileDialog,QDialog
from PyQt5.QtGui import QColor, QFontDatabase, QPixmap
from pyqt5Custom import ToggleSwitch, StyledButton, ImageBox, ColorPicker, ColorPreview, DragDropFile, EmbedWindow, TitleBar, CodeTextEdit, SegmentedButtonGroup, Spinner, Toast
from A1_ import *
import A1_ as network
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dir=['','']
pcv = ''
pointer = 0

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        QFontDatabase.addApplicationFont("data/SFPro.ttf")
        self.setMinimumSize(150, 37)
        self.setGeometry(100, 100, 500, 500)

        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor(255, 255, 255))
        self.setPalette(p)

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.titlebar = TitleBar(self, title="By.WatchMan")
        self.titlebar.setStyleDict({
                "background-color" : (255, 255, 255),
                "font-size" : 17,
                "border-radius": 6,
                "font-family" : "SF Pro Display"
            })

        self.layout.addWidget(self.titlebar, alignment=Qt.AlignTop)

        self.conlyt = QVBoxLayout()
        self.conlyt.setSpacing(0)
        self.conlyt.setContentsMargins(70, 15, 70, 0)
        self.layout.addLayout(self.conlyt)
        h = QLabel("<span style='font-size:58px; font-weight: bold; font-family:SF Pro Display; color:rgb(28,28,30);'>자동 피부암</span>")
        ah = QLabel("<span style='font-size:26px; font-family:SF Pro Display; color:rgb(89,89,92);'>영역 분할 시스템</span>")
        h.setAlignment(Qt.AlignCenter)
        ah.setAlignment(Qt.AlignCenter)
        self.conlyt.addWidget(h)
        self.conlyt.addWidget(ah)

        self.conlyt.addSpacing(90)

        self.btnslyt = QHBoxLayout()
        self.conlyt.addLayout(self.btnslyt)
        self.btnlyt = QVBoxLayout()
        self.btnlyt.setSpacing(16)
        self.btnslyt.addLayout(self.btnlyt)

        self.btnlyt2 = QVBoxLayout()
        self.btnslyt.addLayout(self.btnlyt2)

        self.btn2 = StyledButton("사진 불러오기")
        self.btn2.setFixedSize(170, 54)
        self.btn2.anim_press.speed = 7.3
        self.btn2.setStyleDict({
                "background-color" : (0, 122, 255),
                "border-color" : (0, 122, 255),
                "border-radius" : 7,
                "color" : (255, 255, 255),
                "font-family" : "SF Pro Display",
                "font-size" : 21,
            })
        self.btn2.setStyleDict({
                "background-color" : (36, 141, 255),
                "border-color" : (36, 141, 255)
            }, "hover")
        self.btn2.setStyleDict({
                "background-color" : (130, 190, 255),
                "border-color" : (130, 190, 255),
                "color" : (255, 255, 255),
            }, "press")
        def picture_open():
            global dir, pcv, pointer
            pointer = 0
            dir = QFileDialog.getOpenFileName(self, 'Open file', 'Val_/images')
            dir_basename = os.path.basename(dir[0])
            dir_forder = dir[0].rstrip(dir_basename)
            file_list = os.listdir(dir_forder)
            for a in file_list:
                pointer = pointer + 1
                if (a == dir_basename):
                    print(pointer)
                    break
            image = cv2.imread(dir[0])
            rimage = cv2.resize(image, dsize=(224,224), interpolation= cv2.INTER_AREA)
            cv2.imwrite('save/color.jpg',rimage)
            device_txt = 'cuda:0'
            device = torch.device(device_txt if torch.cuda.is_available() else "cpu")
            num_class = 1
            H = 224;
            W = 224;
            model = network.U_Net(img_ch=3, output_ch=num_class).to(device);
            model.load_state_dict(torch.load('model/newL1_0.008328309282660484_E_899.pth', map_location=device_txt))
            model = model.eval()
            data = MD(path='Val_', H=H, W=W, aug=False);
            x = data.__getitem__(pointer-1)
            inputs = x[0]
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(device)
            outputs = model(inputs.data)
            output = outputs.data
            plt.imsave("save\graylabel.jpg", output[0][0].cpu(), cmap=cm.gray)
            test_image = cv2.imread("save\graylabel.jpg", cv2.IMREAD_COLOR)
            blur = cv2.GaussianBlur(test_image, ksize=(7, 7), sigmaX=0)
            edged = cv2.Canny(blur, 10, 250)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            line = cv.imread('save/color.jpg')
            contours_image = cv2.drawContours(line, contours, -1, (162, 216, 105), 2)
            cv2.imwrite('save/c-line.jpg', contours_image)
            self.close()
            pcv = PictureWindow()
            pcv.show()

        self.btn2.clicked.connect(picture_open)
        self.btnlyt.addWidget(self.btn2, alignment=Qt.AlignTop|Qt.AlignHCenter)

class PictureWindow(QWidget):
    def __init__(self):
        super().__init__()
        QFontDatabase.addApplicationFont("data/SFPro.ttf")

        self.setMinimumSize(150, 37)
        self.setGeometry(100, 100, 890, 610)

        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor(255, 255, 255))
        self.setPalette(p)

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.titlebar = TitleBar(self, title="By.WatchMan")
        self.titlebar.setStyleDict({
                "background-color" : (255, 255, 255),
                "font-size" : 17,
                "border-radius": 6,
                "font-family" : "SF Pro Display"
            })

        self.layout.addWidget(self.titlebar, alignment=Qt.AlignTop)


        self.conlyt = QVBoxLayout()
        self.conlyt.setSpacing(0)
        self.conlyt.setContentsMargins(70, 15, 70, 0)
        self.layout.addLayout(self.conlyt)
        h = QLabel("<span style='font-size:58px; font-weight: bold; font-family:SF Pro Display; color:rgb(28,28,30);'>자동 피부암</span>")
        ah = QLabel("<span style='font-size:26px; font-family:SF Pro Display; color:rgb(89,89,92);'>영역 분할 시스템</span>")
        h.setContentsMargins(100, 0, 0, 0)
        ah.setContentsMargins(103, 0, 0, 0)
        self.conlyt.addWidget(h)
        self.conlyt.addWidget(ah)

        self.conlyt.addSpacing(90)

        self.btnslyt = QHBoxLayout()
        self.conlyt.addLayout(self.btnslyt)
        self.btnlyt = QVBoxLayout()
        self.btnlyt.setSpacing(16)
        self.btnslyt.addLayout(self.btnlyt)

        self.btnlyt2 = QVBoxLayout()
        self.btnslyt.addLayout(self.btnlyt2)

        self.label1 = QLabel('', self)
        self.label1.move(150, 200)
        self.label1.setFixedSize(224,224)
        self.label1.font().setPointSize(500)
        pm = QPixmap(dir[0])
        self.label1.setPixmap(pm.scaled(224,224))
        self.label1.setStyleSheet("color: #4D69E8; border-style: solid; border-width: 2px; border-color: #54A0FF; border-radius: 10px; ")

        self.label2 = QLabel('', self)
        self.label2.move(500, 200)
        self.label2.setFixedSize(224,224)
        self.label2.font().setPointSize(500)
        pm1 = QPixmap('save/c-line.jpg')
        self.label2.setPixmap(pm1.scaled(224,224))
        self.label2.setStyleSheet("color: #4D69E8; border-style: solid; border-width: 2px; border-color: #54A0FF; border-radius: 10px; ")
        self.show()

    def closeEvent(self, event):
        mw.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    pcv = PictureWindow()
    pcv.hide()

    sys.exit(app.exec_())

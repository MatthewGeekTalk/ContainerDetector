import cv2
import numpy as np
from math import *

def cvtBKchar2WHT(img_src):
    if len(img_src.shape) == 3:
        img_src = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
    _, img_src = cv2.threshold(img_src, 255, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    bk_pts_counter = cv2.countNonZero(img_src)
    tt_pts_counter = img_src.shape[0] * img_src.shape[1]
    if tt_pts_counter / bk_pts_counter > 2:
        return 255 - img_src
    else:
        return img_src

def cvtBKchar2WHTs(img_srcs):
    img_grays = []
    img_outputs = []
    black_background = 0
    for img_src in img_srcs:
        if len(img_src.shape) == 3:
            img_src = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
        _, img_src = cv2.threshold(img_src, 255, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        if img_src[img_src.shape[0]-1,0] == 0:  # if the left bottom is black
            black_background = black_background + 1 # count the number of the black background
        img_grays.append(img_src)
    if len(img_grays)/2 - black_background > 0:  # if the background color of most Images are not black
        for img_gray in img_grays:
            img_gray = 255 - img_gray            # convert the color of the img
            img_outputs.append(img_gray)

        return img_outputs
    else:
        return img_grays
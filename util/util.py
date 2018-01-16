import cv2
import numpy as np
from math import *

def cvtBKchar2WHT(img_src):
    if len(img_src.shape) == 3:
        img_src = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)

    _, img_src = cv2.threshold(img_src, 255, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    bk_pts_counter = cv2.countNonZero(img_src)
    tt_pts_counter = img_src.shape[0] * img_src.shape[1]
    # print("bk",bk_pts_counter)
    # print("tt",tt_pts_counter)
    # print("img_src",img_src)

    # cv2.imshow("img_src",img_src)
    if tt_pts_counter / bk_pts_counter > 2:
        # cv2.imshow("img_src2",255 - img_src)
        # cv2.waitKey()
        return 255 - img_src
    else:
        # cv2.imshow("img_src1",img_src)
        # cv2.waitKey()
        return img_src
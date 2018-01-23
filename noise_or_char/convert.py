import sys
import os
import cv2

sys.path.append(os.path.abspath('../tool/'))
from util import util

materials = os.listdir(os.path.abspath('./convert'))
if __name__ == '__main__':
    imgs = []
    for i in range(len(materials)):
        path = os.path.abspath('./convert') + os.path.sep + str(materials[i])
        img = cv2.imread(path)
        # img = util.cvtBKchar2WHT(img)
        imgs.append(img)
    imgs = util.cvtBKchar2WHTs(imgs)
    for i,img in enumerate(imgs):
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        set_path = os.path.abspath('./convert1') + os.path.sep
        cv2.imwrite(set_path + str(i) + '.jpg', img)
    print('finished')
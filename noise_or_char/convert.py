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
        imgs.append(img)
    imgs = util.cvtBKchar2WHTs(imgs)
    for i,img in enumerate(imgs):
        set_path = os.path.abspath('./convert1') + os.path.sep
        cv2.imwrite(set_path + str(i) + '.jpg', img)
    print('finished')
import cv2
from util import rotateImage
import os

def cut_img(img):
    img_cut = img[10:int(img.shape[1]/3),int(img.shape[0]/3):img.shape[0]-5]
    return img_cut
if __name__ == '__main__':
    materials = os.listdir(os.path.abspath('../white'))
    set_path = os.path.abspath('../noise_or_char/cut_white') + os.path.sep
    for i in range(len(materials)):
        path = os.path.abspath('../white') + os.path.sep + str(materials[i])
        img = cv2.imread(path)
        img = rotateImage.docRot(img)
        # blue
        # img_cut = img[0:218, 892:1792]
        # white
        # img_cut = img[60:440, 694:1530]
        img_cut = cut_img(img)
        # img_cut = rotateImage.docRot(img_cut)
        cv2.imwrite(set_path + 'test' + str(i) + '.jpg', img_cut)
        # cv2.imshow('test',img_cut)
        # cv2.waitKey()

import cv2
import rotateImage
import img_cutter
import os

def cut_img(img):
    # blue
    # img_cut = img[84:238,1049:1771]
    # white
    img_cut = img[275:440, 900:1598]
    return img_cut
if __name__ == '__main__':
    materials = os.listdir(os.path.abspath('../util/img/WHITE'))
    set_path = os.path.abspath('../noise_or_char/cut_white') + os.path.sep
    for i in range(len(materials)):
        path = os.path.abspath('../util/img/WHITE') + os.path.sep + str(materials[i])
        img = cv2.imread(path)
        # cv2.imshow('test',img)
        # cv2.waitKey()
        img = rotateImage.docRot(img)
        ratio = [2, 5]
        img_cut = img_cutter.img_cutter(ratio)
        # _, img = img_cut.cut(img)
        img = cut_img(img)
        cv2.imwrite(set_path + 'test' + str(i) + '.jpg', img)


import sys
import os
import cv2
from operator import itemgetter
import numpy as np

sys.path.insert(0, os.path.abspath('./'))
from char_determine_protobuff import CharDetermine

IS_PLATE = [0, 1]
char_dict = {
    'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'B': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'C': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'D': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'E': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'F': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'G': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'H': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'J': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'U': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    'Z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    '0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
          0],
    '1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
          0],
    '2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          0],
    '3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
          0],
    '4': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
          0],
    '5': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
          0],
    '6': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
          0],
    '7': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          0],
    '8': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
          0],
    '9': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1],
}


class PlateRec(object):
    def __init__(self):
        self.path = ""
        self.img = object
        self._chars = []
        self.org_img = object
        self._plate_str = []

    def _charsegment(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # Binaryzation
        _, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        # cv2.imshow("gray", gray)

        # Mser
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        rects = sorted([cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions], key=itemgetter(0, 1, 2, 3))

        rect_temp = set()
        for idx, rect in enumerate(rects):
            flg_continue = 0
            # prune the boxes if the length-width ratio is too large than a normal character
            if rect[2] / rect[3] > 1.5 or rect[3] / rect[2] > 4:
                rect_pre = rect
                continue

            if idx > 0:
                # prune the duplicate boxes
                # if rect == rect_pre:
                #     continue
                if rect[0] == rect_pre[0] and rect[1] == rect_pre[1]:
                    continue

                # prune the boxes which inside another boxes
                for rect_tmp in rect_temp:
                    if rect_tmp[0] <= rect[0] < rect_tmp[0] + rect_tmp[2] \
                            and rect_tmp[1] <= rect[1] < rect_tmp[1] + rect_tmp[3]:
                        flg_continue = 1
                        break
                if flg_continue == 1:
                    continue

                rect_temp.add(rect)
                cv2.rectangle(self.img, rect[0:2], (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)

            rect_pre = rect

            # generate new image as per box
            obj = gray[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            obj = cv2.resize(obj, (28, 28), interpolation=cv2.INTER_CUBIC)
            # set_path = os.path.abspath('../trainingchar1') + os.path.sep
            # cv2.imwrite(set_path + 'test' + str(idx) + '.jpg', obj)
            # print("Num.", len(rect_temp), "Rects Index", idx, "==", rect)
            self._chars.append(obj)
        self.org_img = self.img

    def main(self):
        self._charsegment()
        self.__detect_char(self._chars)

    def __detect_char(self, chars):
        plate_string = ""
        char_determine = CharDetermine()

        if len(chars) != 0:
            imgs, labels = char_determine.main(chars)
            if len(imgs) == len(labels):
                for i in range(len(imgs)):
                    for key, value in char_dict.items():
                        if value == labels[i]:
                            plate_string += key
            self._plate_str.append(plate_string)

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, img):
        self._img = img

    @property
    def img_con_sobel(self):
        return self._img_con_sobel

    @property
    def plates_sobel(self):
        return self._plates_sobel

    @property
    def regions_sobel(self):
        return self._regions_sobel

    @property
    def img_con_color(self):
        return self._img_con_color

    @property
    def plates_color(self):
        return self._plates_color

    @property
    def regions_color(self):
        return self._regions_color

    @property
    def plate_with_no(self):
        return self._plate_with_no

    @property
    def plate_string(self):
        return self._plate_str

    @property
    def plates_sobel_ori(self):
        return self._plates_sobel_ori

    @property
    def plates_color_ori(self):
        return self._plates_color_ori


if __name__ == '__main__':

    print('Image path: %s' % str(os.path.abspath('../Material')).replace('\\', '\\\\'))
    path = input('Please input your image path:')

    img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    plate_rec = PlateRec()
    plate_rec.img = img

    plate_rec.main()

    for plate in plate_rec.plates_sobel_ori:
        plate_rec.print_plate(plate)

        # for plate in plate_rec.plates_color_ori:
        #     plate_rec.print_plate(plate)

        # path2 = input('Please input your saving path:')
        # plate_rec.save_plate(path2, plate)

    plate_rec.print_plate(plate_rec.img_con_sobel)
    plate_rec.print_plate(plate_rec.plates_color)
    # plate_rec.print_plate(plate_rec.img_con_color)
    print(plate_rec.plate_string)

    # for plate in plate_rec.plate_with_no:
    #     for char in plate['value']:
    #         plate_rec.print_plate(char)

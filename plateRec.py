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
        # gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # ret, img_thre = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        # mser = cv2.MSER_create()
        # regions, bboxes = mser.detectRegions(gray)
        # boxs = []
        # box_safe = []
        # point = np.array([0, 0])
        # index = 1
        # tolerance = 30
        # # Use aspect ratio to delete bounding box
        # for p in bboxes:
        #     p[0] = p[0] - 2
        #     p[1] = p[1] + 2
        #     p[2] = p[2] + 2
        #     p[3] = p[3] + 2
        #     box = [[p[0], p[1]], [p[0] + p[2], p[1]], [p[0], p[1] + p[3]], [p[0] + p[2], p[1] + p[3]]]
        #     if p[3] > 2.5 * p[2] or p[3] < 1.5 * p[2]:
        #         continue
        #     boxs.append(box)
        # boxs.sort()
        # # Use first coordinate to delete bounding box
        # for p in boxs:
        #     if index == 1:
        #         point = [p[0][0], p[0][1]]
        #     if index >= 2 and p[0][0] - tolerance <= point[0] <= p[0][0] + tolerance and p[0][1] - tolerance <= point[
        #         1] <= p[0][1] + tolerance:
        #         continue
        #     box_safe.append(p)
        #     point = [p[0][0], p[0][1]]
        #     index = index + 1
        # # Cut bounding box from input pic and save
        # for box in box_safe:
        #     x1 = box[0][0]
        #     x2 = box[1][0]
        #
        #     y1 = box[0][1]
        #     y2 = box[2][1]
        #     char = img_thre[y1:y2, x1:x2]
        #     char = cv2.resize(char, (28, 28), interpolation=cv2.INTER_CUBIC)
        #     # set_path = os.path.abspath('../trainingchar1') + os.path.sep
        #     # cv2.imwrite(set_path + str(i) + '.jpg', char)
        #     points = np.array(
        #         [[box[0][0], box[0][1]], [box[2][0], box[2][1]], [box[3][0], box[3][1]], [box[1][0], box[1][1]]])
        #     cv2.polylines(self.img, np.int32([points]), 1, (0, 255, 0))
        #     self._chars .append(char)
        # Read image
        # img = cv2.imread(self.img, cv2.CAP_OPENNI_GRAY_IMAGE)
        # Convert to gray
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

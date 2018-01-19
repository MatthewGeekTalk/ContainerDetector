import cv2
import numpy as np

class img_cutter:
    # Constructor
    def __init__(self, ratio):
        self.ratio = ratio

    # Cut image according to ratio
    def cut(self, img):

        imgs = []

        # Weight & Height
        w = np.round(img.shape[1] / self.ratio[0])
        h = np.round(img.shape[0] / self.ratio[1])

        for i in range(0, self.ratio[0]):
            x1 = int(i*w)
            if i < self.ratio[0] - 1:
                x2 = int((i+1)*w)
            else:
                x2 = img.shape[1]

            for j in range(0, self.ratio[1]):
                y1 = int(j*h)
                if j < self.ratio[1] - 1:
                    y2 = int((j+1)*h)
                else:
                    y2 = img.shape[0]

                imgs.append(img[y1:y2, x1:x2])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.imshow('img', img)
        return imgs


    # Main
    def main(self):
        img = cv2.imread('Container.jpg')
        imgs = self.cut(img)
        i = 0
        for img_1 in imgs:
          i = i + 1
          name = 'img' + str(i)
          cv2.imshow(name, img_1)
          cv2.waitKey(0)

if __name__ == '__main__':

    ratio = [2,2]

    cutter = img_cutter(ratio)
    cutter.main()

import cv2
import os

materials = os.listdir(os.path.abspath('../noise_or_char/positive_28x28'))

if __name__ == '__main__':
    for i in range(len(materials)):
        path = os.path.abspath('../noise_or_char/positive_28x28') + os.path.sep + str(materials[i])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img_thre = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        # img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
        set_path = os.path.abspath('../noise_or_char/positive_b') + os.path.sep
        cv2.imwrite(set_path + str(i) + '.jpg', img_thre)
    print('finished')
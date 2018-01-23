import cv2

img = cv2.imread('lena.jpg')
bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gary = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY)

cv2.imshow('test',gary)
cv2.waitKey()
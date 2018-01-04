import numpy as np
import cv2
import os

img = cv2.imread('container.jpg');
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vis = img.copy()
mser = cv2.MSER_create()
regions, bboxes = mser.detectRegions(gray)
rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
boxs = []
box_safe = []
point=np.array([0,0])
index = 1
tolerance = 30
# Use aspect ratio to delete bounding box
for p in bboxes:
  box = [[p[0], p[1]], [p[0] + p[2], p[1]], [p[0], p[1] + p[3]], [p[0] + p[2], p[1] + p[3]]]
  if p[3] > 3 * p[2] and p[3] < 4.5 * p[2]: # for number 1
    boxs.append(box)
    continue
  if p[3] > 2.5 * p[2] or p[3] < 1.5 * p[2]:
    continue
  boxs.append(box)
boxs.sort()
# Use first coordinate to delete bounding box
for p in boxs:
  if index == 1:
    point = [p[0][0], p[0][1]]
  if index >= 2 and p[0][0] - tolerance <= point[0] <= p[0][0] + tolerance and p[0][1] - tolerance <= point[1] <= p[0][1] + tolerance:
    continue
  box_safe.append(p)
  point = [p[0][0], p[0][1]]
  index = index + 1
i = 1
# Cut bounding box from input pic and save 
for box in box_safe:
  x1 = box[0][0]
  x2 = box[1][0]

  y1 = box[0][1]
  y2 = box[2][1]
  char = img[y1:y2, x1:x2]
  set_path = os.path.abspath('trainingchar1') + os.path.sep
  cv2.imwrite(set_path + 'test' + str(i) + '.jpg', char)
  print(i,x1, x2, y1, y2)
  points = np.array([[box[0][0], box[0][1]], [box[2][0], box[2][1]], [box[3][0], box[3][1]], [box[1][0], box[1][1]]])
  cv2.polylines(img, np.int32([points]), 1, (0, 255, 0))
  i = i + 1
  
cv2.imshow('img', img)
cv2.waitKey()

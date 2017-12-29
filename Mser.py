import cv2

img = cv2.imread('container.jpg');
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vis = img.copy()
mser = cv2.MSER_create()
regions, bboxes = mser.detectRegions(gray)
rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
boxs = []
for p in bboxes:
  box = [ [ p[0], p[1] ],[ p[0]+p[2] ,p[1] ],[ p[0], p[1]+p[3] ],[ p[0]+p[2], p[1]+p[3] ] ]
  boxs.append(box)
i = 1
for box in boxs:
  x1 = box[0][0]
  x2 = box[1][0]

  y1 = box[0][1]
  y2 = box[2][1]
  char = img[y1:y2, x1:x2]
  set_path = os.path.abspath('../trainingchar1') + os.path.sep
  # cv2.imwrite(set_path + 'test' + str(i) + '.jpg', char)
  i = i + 1
for rect in rects:
  cv2.rectangle(img, rect[0:2], (rect[0]+rect[2],rect[1]+rect[3]), (0, 255, 0), 1)
cv2.imshow('img', img)
cv2.waitKey()

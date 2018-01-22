import copy

class groupBox(object):

    yoffset = [0.99, 1.01]
    highoffset = [0.95, 1.05]
    lineYoffset = [0.95, 1.05]

    # Constructor
    def __init__(self, boxes):
        self.boxes = boxes
        self.containerID = []

    def calculateHighMid(self, Box):
        return ( Box[2][1] + Box[0][1] ) / 2

    def calculateHigh(self, Box):
        return Box[2][1] - Box[0][1]

    def calculateCenterPoint(self, Boxes):
        x = ( Boxes[0][0][0] + Boxes[0][2][0] + Boxes[len(Boxes) - 1][1][0] + Boxes[len(Boxes) - 1][3][0] ) / 4
        y = ( Boxes[0][0][1] + Boxes[0][2][1] + Boxes[len(Boxes) - 1][1][1] + Boxes[len(Boxes) - 1][3][1] ) / 4
        return [x, y]

    def generate_id_boxes(self):
        idLine   = []
        line_box = []
        # check which box should be one line
        self.boxes.sort()
        while self.boxes.__len__() != 0:
            for box in self.boxes:
                line_box.clear()
                line_box.append(box)
                for sameLine in self.boxes:

                    if self.calculateHighMid(box) * self.yoffset[0] < self.calculateHighMid(sameLine) and self.calculateHighMid(
                            box) * self.yoffset[1] > self.calculateHighMid(sameLine):  # 0.9 * y < same_line y < 1.1 * y
                        if self.calculateHigh(box) * self.highoffset[0] < self.calculateHigh(sameLine) and self.calculateHigh(
                                box) * self.highoffset[1] > self.calculateHigh(sameLine):  # 0.95 * High < same_line high < 1.05 * High
                            if box == sameLine:
                                continue
                            line_box.append(sameLine)
                            box = sameLine

                for delete in line_box:
                    self.boxes.remove(delete)  # delete the whole line boxes
                idLine.append(copy.deepcopy(line_box))
                self.boxes.sort()
                break

        # find the unique line
        for boxes in idLine:
            if boxes.__len__() >= 3: # find the first 4 character
                boxNumber = boxes.__len__()
                for secondBoxes in idLine:
                    if boxes == secondBoxes:
                        continue
                    if secondBoxes.__len__() >= 4: # only retrieve the line contains 4 boxes
                        firstCenter = self.calculateCenterPoint(boxes)
                        secondCenter = self.calculateCenterPoint(secondBoxes)
                        if secondCenter[0] > firstCenter[0] and firstCenter[1] * self.lineYoffset[0] < secondCenter[1] and firstCenter[1] * self.lineYoffset[1] > secondCenter[1]:
                            boxes.extend(secondBoxes)
                            break
                if boxNumber != boxes.__len__() or boxNumber >= 9:
                    containerID = copy.deepcopy(boxes)
                    break

        # find the last one
        for boxes in idLine:
            if boxes.__len__() == 1:
                firstCenter = self.calculateCenterPoint(containerID)
                secondCenter = self.calculateCenterPoint(boxes)
                if secondCenter[0] > firstCenter[0] and firstCenter[1] * self.lineYoffset[0] < secondCenter[1] and firstCenter[1] * self.lineYoffset[1] > secondCenter[1]:
                    containerID.extend(boxes)
                    break
        return containerID

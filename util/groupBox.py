import copy

class groupBox(object):

    yoffset = [0.99, 1.01]
    highoffset = [0.95, 1.05]

    # Constructor
    def __init__(self, boxes):
        self.boxes = boxes

    def calculateHighMid(self, Box):
        return ( Box[2][1] + Box[0][1] ) / 2

    def calculateHigh(self, Box):
        return Box[2][1] - Box[0][1]

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

        return idLine
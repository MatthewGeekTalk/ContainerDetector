from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageOps
import random
import sys
import os
import numpy as np

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FONTS = ["Rama", "Franklin", "Gilli", "TCC", "Tahoma"]
FONT_FILES = {"Rama": "RamaGothicE_SemiBold.otf",
              "Franklin": "FRAMDCN.TTF",
              "TCC": "TCCB____.TTF",
              "Tahoma": "tahoma.ttf",
              "Gilli": "GILC____.TTF",
              }
BLACK = (20, 20, 20)
WHITE = (250, 250, 250)
GREY = (70, 70, 70)

POSITIONS = {1: {"Rama": [(1110, 115), (1366, 115), (1678, 120)],
                 "Franklin": [(1080, 115), (1360, 119), (1672, 120)],
                 "Gilli": [(1080, 110), (1380, 119), (1678, 116)],
                 "TCC": [(1080, 115), (1375, 125), (1675, 124)],
                 "Tahoma": [(1080, 115), (1375, 125), (1675, 124)]},
             6: {"Rama": [(850, 240), (1080, 240), (1416, 236)],
                 "Franklin": [(810, 240), (1080, 240), (1406, 236)],
                 "Gilli": [(850, 240), (1080, 240), (1414, 236)],
                 "TCC": [(820, 240), (1080, 240), (1410, 240)],
                 "Tahoma": [(805, 245), (1080, 240), (1406, 236)]}}
OFFSET_1 = {1: {"Rama": 8,
                "Franklin": 0,
                "Gilli": 3,
                "TCC": 3,
                "Tahoma": 0},
            6: {"Rama": 12,
                "Franklin": 0,
                "Gilli": 12,
                "TCC": 3,
                "Tahoma": 3}}
HEIGHT = {1: {"Rama": 95,
              "Franklin": 95,
              "Gilli": 95,
              "TCC": 95,
              "Tahoma": 75},
          6: {"Rama": 90,
              "Franklin": 90,
              "Gilli": 90,
              "TCC": 90,
              "Tahoma": 80}}
COLOUR = {1: WHITE,
          6: BLACK}
ROTATION = {1: 0,
            6: 3}


def generate_first_letters():
    return "{}{}{}U".format(
        random.choice(LETTERS),
        random.choice(LETTERS),
        random.choice(LETTERS))


def generate_numbers():
    return "{}{}{}{}{}{}".format(
        random.choice(DIGITS),
        random.choice(DIGITS),
        random.choice(DIGITS),
        random.choice(DIGITS),
        random.choice(DIGITS),
        random.choice(DIGITS))


def generate_checksum():
    return random.choice(DIGITS)


def choose_font():
    return random.choice(FONTS)


def generate_img(img, t, font_name):
    firstLet = generate_first_letters()
    num = generate_numbers()
    checksum = generate_checksum()
    # get positions for this image and font
    pos = POSITIONS[t][font_name]
    font = ImageFont.truetype(FONT_FILES[font_name], HEIGHT[t][font_name])
    if checksum == "1":
        # Some fonts have a smaller "1" than other numbers, add offset to place 1 in middle of the box
        posCheck = (pos[2][0] + OFFSET_1[t][font_name], pos[2][1])
    else:
        posCheck = pos[2]
    # print(posCheck)
    rot = ROTATION[t]
    # check whether rotating the letters is necessary
    if rot == 0:
        # No rotation necessary, just add text onto the image
        colour = COLOUR[t]
        draw = ImageDraw.Draw(img)
        draw.text(pos[0], firstLet, fill=colour, font=font)
        draw.text(pos[1], num, colour, font=font)
        draw.text(posCheck, checksum, colour, font=font)
    else:
        # Rotation needed, draw text onto separate image, roatate and then merge images
        width, height = img.size
        txt = Image.new(mode='L', size=(width, height), color=0)
        draw = ImageDraw.Draw(txt)
        draw.text(pos[0], firstLet, fill=255, font=font)
        draw.text(pos[1], num, fill=255, font=font)
        draw.text(pos[2], checksum, 255, font=font)
        w = txt.rotate(rot, expand=1)
        img.paste(ImageOps.colorize(w, (0, 0, 0), GREY), (0, 0), w)
    return img, firstLet, num, checksum


def generate_imgs(numImages, imgPath, savePath, t):
    for i in range(numImages):
        font = choose_font()
        img = Image.open(imgPath)
        img, let, num, checksum = generate_img(img, t, font)
        img.save(savePath + let + "_" + num + "_" + checksum + ".jpg")

if __name__ == '__main__':
    imgpath = os.path.abspath('./') + os.path.sep + 'model1a.jpg'
    savePath = os.path.abspath('./img') + os.path.sep
    generate_imgs(2, imgpath, savePath, 1)

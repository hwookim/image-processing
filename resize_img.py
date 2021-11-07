import sys
import cv2


def run():
    filename = sys.argv[-1]
    img = cv2.imread(filename, 0)
    name = filename.split('.')[0]

    height, width = img.shape
    height_const = height % 2
    width_const = width % 2

    result = cv2.resize(img, (height - height_const, width - width_const))
    cv2.imwrite(name + '_resized.png', result)


run()

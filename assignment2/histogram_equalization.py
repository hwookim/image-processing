import cv2
import numpy as np


def run():
    in_image = cv2.imread('dgu_night.png', 0)
    out_image = histogram_equalization(in_image)
    cv2.imwrite('dgu_equalize.png', out_image)


def histogram_equalization(img):
    height, width = img.shape
    level = np.zeros(256)

    for i in img.ravel():
        level[i] += 1

    cumulated_level = cumulate(level)
    nomalized_level = nomalize(cumulated_level, img.size)

    result = np.zeros_like(img)
    for x in range(width):
        for y in range(height):
            result[y, x] = nomalized_level[img[y, x]]
    print(result)
    return result


def cumulate(data):
    cumulated = np.zeros_like(data)
    cumulated[0] = data[0]
    for i in range(1, cumulated.size):
        cumulated[i] = cumulated[i - 1] + data[i]

    return cumulated


def nomalize(data, size):
    return np.round((data) * 255 / size)


run()

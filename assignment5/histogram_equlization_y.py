import sys
import cv2
import numpy as np


def run():
    filename = sys.argv[-1]
    img = cv2.imread(filename)
    name = filename.split('.')[0]

    result = equalize_histogram_y(img)
    cv2.imwrite(name + "_equalized.png", result)


def equalize_histogram_y(img: np.ndarray) -> np.ndarray:
    y, cr, cb = convert_to_YCrCb(img)
    equalized_y = equalize_histogram(y)
    equalized_ycrcb = (np.dstack((equalized_y, cr, cb))).astype(np.uint8)
    result = cv2.cvtColor(equalized_ycrcb, cv2.COLOR_YCR_CB2BGR)
    return result


def convert_to_YCrCb(img: np.ndarray):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(ycrcb)
    return y, cr, cb


def equalize_histogram(img: np.ndarray) -> np.ndarray:
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
    return result


def cumulate(data: np.ndarray) -> np.ndarray:
    cumulated = np.zeros_like(data)
    cumulated[0] = data[0]
    for i in range(1, cumulated.size):
        cumulated[i] = cumulated[i - 1] + data[i]

    return cumulated


def nomalize(data: np.ndarray, size: int) -> np.ndarray:
    return np.round((data) * 255 / size)


run()
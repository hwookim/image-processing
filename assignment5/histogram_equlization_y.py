import sys
import cv2
import numpy as np


def run():
    filename = sys.argv[1]
    s = float(sys.argv[2])

    img = cv2.imread(filename)
    name = filename.split('.')[0]

    result = equalize_histogram_y(img, s)
    cv2.imwrite(name + "_equalized_" + str(s) + ".png", result)


def equalize_histogram_y(img: np.ndarray, s: float) -> np.ndarray:
    y, cr, cb = convert_to_YCrCb(img)
    equalized_y = equalize_histogram(y)

    trans_equalized_y = np.transpose(
        np.tile(equalized_y, (3, 1, 1)), (1, 2, 0))
    trans_y = np.transpose(np.tile(y, (3, 1, 1)), (1, 2, 0))
    result = trans_equalized_y * \
        (np.divide(img, trans_y, where=trans_y != 0) ** s)

    return np.clip(result, 0, 255)


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

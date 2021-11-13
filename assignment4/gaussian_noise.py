import sys
import cv2
import numpy as np


def run():
    filename = sys.argv[-1]
    img = cv2.imread(filename, 0)
    name = filename.split('.')[0]

    noise_cnt = 12
    noised_imgs = np.array(range(noise_cnt), object)
    for i in range(noise_cnt):
        noised_imgs[i] = add_gaussian_noise(img, 20)
        cv2.imwrite(name + "_noise_" + str(i) + ".png", noised_imgs[i])

    avarage_img = average_imgs(noised_imgs)
    cv2.imwrite(name + "_noise_removed.png", avarage_img)


def add_gaussian_noise(img: np.ndarray, strength: int) -> np.ndarray:
    height, width = img.shape
    noised_img = np.zeros((height, width), np.float64)
    for h in range(height):
        for w in range(width):
            noise = strength * np.random.normal()
            noised_img[h][w] = img[h][w] + noise
    return noised_img


def average_imgs(imgs: np.ndarray) -> np.ndarray:
    height, width = imgs[0].shape
    length = len(imgs)
    avarage_img = np.zeros((height, width), np.float64)
    for h in range(height):
        for w in range(width):
            sum = 0
            for i in range(length):
                sum += imgs[i][h][w]
            if sum / length > 255:
                avarage_img[h][w] = 255
            else:
                avarage_img[h][w] = sum / length
    return avarage_img


run()

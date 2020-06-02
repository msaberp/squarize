import cv2 as cv
import numpy as np


def convolve(src, kernel, iterations=1):
    output = src.copy()
    for it in range(iterations):
        convolved = cv.filter2D(output, -1, kernel, borderType=cv.BORDER_CONSTANT)
        convolved = cv.threshold(convolved, 180, 255, cv.THRESH_BINARY)[1]
        output = cv.bitwise_or(output, convolved)
    return output
    
    
def squarize(src, iterations=1):
    kernel_ul = np.array([[0, 0, 0], [0, 0, 1/3], [0, 1/3, 1/3]])
    kernel_ur = np.array([[0, 0, 0], [1/3, 0, 0], [1/3, 1/3, 0]])
    kernel_dl = np.array([[0, 1/3, 1/3], [0, 0, 1/3], [0, 0, 0]])
    kernel_dr = np.array([[1/3, 1/3, 0], [1/3, 0, 0], [0, 0, 0]])
    kernels = [kernel_ul, kernel_ur, kernel_dl, kernel_dr]
    conv = src.copy()
    for kernel in kernels:
        conv = convolve(conv, kernel, iterations)
    return conv


def inv_squarize(src, iterations=1):
    return cv.bitwise_not(squarize(cv.bitwise_not(src), iterations))


def squarize_and_close(src, kernel, iterations=1):
    squared = squarize(src, iterations)
    closed = cv.morphologyEx(squared, cv.MORPH_CLOSE, kernel)
    return closed


def inv_squarize_and_open(src, kernel, iterations=1):
    inv_squared = inv_squarize(src, iterations)
    opened = cv.morphologyEx(inv_squared, cv.MORPH_OPEN, kernel)
    return opened


if __name__ == "__main__":
    iters = 1
    img = cv.imread("image_1.png", 0)
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    _kernel = np.ones((3, 3))

    inv_sq = inv_squarize_and_open(img, _kernel, iters)
    sq = squarize_and_close(inv_sq, _kernel, iters)
    inv_sq = inv_squarize_and_open(sq, _kernel, iters)

    cv.imwrite("sq.png", sq)
    cv.imwrite("inv_sq.png", inv_sq)


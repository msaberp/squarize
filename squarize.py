import cv2 as cv
import numpy as np


def convolve(src, kernel, iterations=1):
    output = src.copy()
    for it in range(iterations):
        conv = cv.filter2D(output, -1, kernel)
        _, thr = cv.threshold(conv, 180, 255, cv.THRESH_BINARY)
        output = cv.threshold(thr + src, 180, 255, cv.THRESH_BINARY)[1]
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
    for it in range(iterations):
        squared = squarize(src)
        src = cv.morphologyEx(squared, cv.MORPH_CLOSE, kernel)
    return src


def inv_squarize_and_open(src, kernel, iterations=1):
    for it in range(iterations):
        inv_squared = inv_squarize(src)
        src = cv.morphologyEx(inv_squared, cv.MORPH_OPEN, kernel)
    return src


if __name__ == "__main__":
    iters = 1
    img = cv.imread("module.png", 0)
    _kernel = np.ones((3, 3))
    sq_and_close = squarize_and_close(img, _kernel, iters)
    cv.imwrite("module_sq_close.png", sq_and_close)
    inv_sq_and_open = inv_squarize_and_open(sq_and_close, _kernel, iters)
    cv.imwrite("module_inv_sq_open.png", inv_sq_and_open)


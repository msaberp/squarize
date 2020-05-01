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
    
if __name__ == "__main__":
    iters = 20
    img = cv.imread("circle.png", 0)
    cv.imwrite("square.png", squarize(img, iters))
    

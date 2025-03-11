# import Opencv
import cv2

# import Numpy
import numpy as np

# read a image using imread
img = cv2.imread('example_image.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

# creating a Histograms Equalization
# of a image using cv2.equalizeHist()
equ = cv2.equalizeHist(img)

# stacking images side-by-side
res = np.hstack((img, equ))

# show image input vs output
cv2.imshow('processed_image', res)

cv2.waitKey(0)
cv2.destroyAllWindows()

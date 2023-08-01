import cv2

image1 = cv2.imread('../../image1_60_left.jpg')
image2 = cv2.imread('../../image1_60_right.jpg')

cv2.imshow('image1', image1)
cv2.imshow('image2', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
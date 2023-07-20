import cv2
import matplotlib.pyplot as plt
import time
image1= cv2.imread("image1_60_left.jpg",flags=cv2.IMREAD_GRAYSCALE)
image2= cv2.imread("image1_60_right.jpg",flags=cv2.IMREAD_GRAYSCALE)

BRISK = cv2.BRISK_create()
time_start=time.time()
keypoints1, descriptors1 = BRISK.detectAndCompute(image1, None)
keypoints2, descriptors2 = BRISK.detectAndCompute(image2, None)

# create BFMatcher object
BFMatcher = cv2.BFMatcher(normType = cv2.NORM_HAMMING,
                         crossCheck = True)

# Matching descriptor vectors using Brute Force Matcher
matches = BFMatcher.match(queryDescriptors = descriptors1,
                          trainDescriptors = descriptors2)
time_end=time.time()
# Sort them in the order of their distance
matches = sorted(matches, key = lambda x: x.distance)

# Draw first 15 matches
output = cv2.drawMatches(img1 = image1,
                        keypoints1 = keypoints1,
                        img2 = image2,
                        keypoints2 = keypoints2,
                        matches1to2 = matches[:1000],
                        outImg = None,
                        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(output)
plt.show()
print(time_end-time_start)
import cv2
from matplotlib import pyplot as plt

import numpy as np

def plot_image(img, figsize_in_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
    
def plot_images(imgs, figsize_in_inches=(5,5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
image_file1 = 'yolov7/IMG_1981.jpeg'
image1=cv2.imread(image_file1)
print(image1.shape)
height, width, channels = image1.shape
# print(height/2,width/2)
# half = width // 2
# image1_50_left = image1[:, :half]
# image1_50_right = image1[:, half:] 
# print(image1_50_left.shape, image1_50_right.shape)
# cv2.imshow('image1_50_left',image1_50_left)
# cv2.imshow('image1_50_right',image1_50_right)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('image1_50_left.jpg',image1_50_left)
# cv2.imwrite('image1_50_right.jpg',image1_50_right)
sixty_percent = int(width * 0.6)
image1_60_left = image1[:, :sixty_percent]
image1_60_right = image1[:, width-sixty_percent:]
print(image1_60_left.shape, image1_60_right.shape)
cv2.imshow('image1_60_left',image1_60_left)
cv2.imshow('image1_60_right',image1_60_right)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image1_60_left.jpg',image1_60_left)
cv2.imwrite('image1_60_right.jpg',image1_60_right)


# left_image = cv2.imread('image1_60_left.jpg')
# right_image = cv2.imread('image1_60_right.jpg')
# stitching_image=cv2.Stitcher_create()
# status,stitched_image=stitching_image.stitch((left_image,right_image))

# if status==0:
#     cv2.imshow('stitched_image',stitched_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite('stitched_image.jpg',stitched_image)
#     print('image stitched', stitched_image.shape)
# else:
#     print("Error")
#     cv2.destroyAllWindows()

# import stitching
# settings = {"detector": "sift", "confidence_threshold": 0.2}
# stitcher = stitching.Stitcher(**settings)
# panorama = stitcher.stitch(['image1_60_left.jpg', 'image1_60_right.jpg'])
# plot_image(panorama)

# image_file2 = 'traffic_image_ultrawide_2.jpg'
# image2=cv2.imread(image_file2)
# print('image2 shape',image2.shape)
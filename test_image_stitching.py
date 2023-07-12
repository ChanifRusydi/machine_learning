import cv2
import time
import numpy as np
# def open_camera(index):
#     cap = cv2.VideoCapture(index)
#     return cap

# def main():
#     # print OpenCV version
#     print("OpenCV version: " + cv2.__version__)

#     # Get camera list
#     # device_list = device.getDeviceList()
#     # index = 0

#     # for camera in device_list:
#     #     print(str(index) + ': ' + camera[0])
        
#     #     index += 1

#     # last_index = index - 1

#     # if last_index < 0:
#     #     print("No device is connected")
#     #     return

#     # Select a camera
#     # camera_number = select_camera(last_index)
    
#     # Open camera
#     cap1 = open_camera(0)
#     cap2= open_camera(1)
#     print("1st webcam opened" if cap1.isOpened() else "1st webcam failed to open")
#     print("2nd webcam opened" if cap2.isOpened() else "2nd webcam failed to open")

#     if cap1.isOpened() or cap2.isOpened():
#         width1 = cap1.get(3) # Frame Width
#         height1 = cap1.get(4) # Frame Height
#         print('Default width: ' + str(width1) + ', height: ' + str(height1))
#         # cap1.set(cv2.CAP_PROP_FRAME_WIDTH(640))
#         # cap2.set(cv2.CAP_PROP_FRAME_HEIGHT(480))

#         while True:
#             ret1, frame1 = cap1.read()
#             ret2, frame2 = cap2.read()
#             if ret1 == False or ret2 == False:
#                 print("No camera")
#             else:
                
#                 # cv2.save('frame.jpg',frame)
#                 # cv2.save('frame1.jpg',frame1)
#                 stitching=cv2.Stitcher.create()
#                 status,stitched=stitching.stitch((frame1,frame2))
#                 if status==0:
#                     print("Success stitching at" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#                     cv2.imshow('stitched',stitched)
#                 else:
#                     current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#                     print("Error at " + current_time)
#             # Display the resulting frame
#                 side_by_side = cv2.hconcat([frame1, frame2])
#                 cv2.imshow('side_by_side', side_by_side)

#                 # key: 'ESC'
#                 key = cv2.waitKey(20)
#                 if key == 27:
#                     break

#         cap.release()
#         cap1.release() 
#         cv2.destroyAllWindows() 

# if __name__ == "__main__":
#     main()

# cv2.xfeatures2d_SURF.create()
# WARP_CHOICES='plane'
# ESTIMATOR_CHOICES='homography'

# def main():
#     # print OpenCV version
#     print("OpenCV version: " + cv2.__version__)
def stitch_images(images, warp_type='plane', estimator_type='homography'):
    # Convert the images to grayscale
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # Detect keypoints and extract features using SIFT
    sift = cv2.SIFT_create()
    keypoints = [sift.detect(img, None) for img in gray_images]
    descriptors = [sift.compute(img, kp)[1] for img, kp in zip(gray_images, keypoints)]

    # Match features between the images
    matcher = cv2.BFMatcher()
    matches = [matcher.match(desc1, desc2) for desc1, desc2 in zip(descriptors[:-1], descriptors[1:])]

    # Compute the homography matrices using RANSAC
    homographies = []
    for match in matches:
        src_pts = np.float32([keypoints[0][m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[1][m.trainIdx].pt for m in match]).reshape(-1, 1, 2)
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        homographies.append(homography)

    # Warp the images using the homography matrices
    if warp_type == 'plane':
        warper = cv2.PlanarWarper()
    elif warp_type == 'cylindrical':
        warper = cv2.CylindricalWarper()
    elif warp_type == 'spherical':
        warper = cv2.SphericalWarper()
    else:
        raise ValueError(f'Invalid warp type: {warp_type}')

    if estimator_type == 'homography':
        estimator = cv2.detail_HomographyBasedEstimator()
    elif estimator_type == 'affine':
        estimator = cv2.detail_AffineBasedEstimator()
    else:
        raise ValueError(f'Invalid estimator type: {estimator_type}')

    stitcher = cv2.createStitcher()
    stitcher.setWarper(warper)
    stitcher.setEstimator(estimator)
    status, stitched = stitcher.stitch(images[:-1], homographies)

    if status != cv2.Stitcher_OK:
        raise RuntimeError(f'Stitching failed with status {status}')

    return stitched

# Example usage
if __name__ == '__main__':
    # Load the images
    image1=cv2.imread('image1_60_left.jpg')
    image2=cv2.imread('image1_60_right.jpg')
    images = [image1, image2]

    # Stitch the images
    stitched = stitch_images(images)

    # Display the result
    cv2.imshow('Stitched Image', stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
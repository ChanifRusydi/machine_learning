import cv2
import argparse
import time

import platform
import sys

parser = argparse.ArgumentParser(description='Image Stitching')
parser.add_argument('--image', help='image stitching on image files')
parser.add_argument('--webcam', default=0, type=int, help='image stitching on live video from webcam')


if platform.system() == 'Windows':
    import device

elif platform.system() == 'Linux':
    print('Linux')
elif platform.system() == 'Darwin':
    print('Mac')


# def select_camera(last_index, camera_name):
#     for camera_name_index in camera_name:
#         if camera_name_index == "Logi C270 WebCam":
#             camera_index= last_index


def open_camera(index):
    cap = cv2.VideoCapture(index)
    return cap

def main():
    # print OpenCV version
    print("OpenCV version: " + cv2.__version__)

    # Get camera list
    device_list = device.getDeviceList()
    index = 0

    for camera in device_list:
        print(str(index) + ': ' + camera[0])
        
        index += 1

    last_index = index - 1

    if last_index < 0:
        print("No device is connected")
        return

    # Select a camera
    # camera_number = select_camera(last_index)
    
    # Open camera
    cap = open_camera(0)
    cap1 = open_camera(1)

    if cap.isOpened():
        width = cap.get(3) # Frame Width
        height = cap.get(4) # Frame Height
        print('Default width: ' + str(width) + ', height: ' + str(height))

        while True:
            
            ret, frame = cap.read()
            ret1, frame1 = cap1.read()
            if ret == False or ret1 == False:
                print("No camera")
                break
            # else:
            #     stitching=cv2.Stitcher.create()
            #     status,frame2=stitching.stitch((frame,frame1))
            #     if status==0:
            #         cv2.imshow('frame',frame2)
            #     else:
            #         print("Error")
            #         break
            # Display the resulting frame
            # side_by_side = cv2.hconcat([frame, frame1])
            # cv2.imshow('frame', side_by_side)

            # key: 'ESC'
            key = cv2.waitKey(20)
            if key == 27:
                break

        cap.release()
        cap1.release() 
        cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()

# import cv2
# image1=cv2.imread('image1.jpg')
# image2=cv2.imread('image2.jpg')
# height1, width1 = image1.shape[:2]
# height2, width2 = image2.shape[:2]
# print(height1, width1, height2, width2)
# stitching=cv2.Stitcher.create()
# status,stitched_image=stitching.stitch((image1,image2))
# if status==0:
#     cv2.imshow('frame',stitched_image)
#     cv2.imwrite('stitched_image.jpg',stitched_image)
# else:
#     print(status)

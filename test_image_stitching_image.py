import cv2
import time
def open_camera(index):
    cap = cv2.VideoCapture(index)
    return cap

def main():
    # print OpenCV version
    print("OpenCV version: " + cv2.__version__)
    # Get camera list
    # device_list = device.getDeviceList()
    # index = 0
    # for camera in device_list:
    #     print(str(index) + ': ' + camera[0])
    #     index += 1
    # last_index = index - 1

    # if last_index < 0:
    #     print("No device is connected")
    #     return
    # Select a camera
    # camera_number = select_camera(last_index)
    # Open camera
    cap1 = open_camera(0)
    cap2 = open_camera(1)
    if cap1.isOpened() or cap2.isOpened():
        width1 = cap1.get(3) # Frame Width
        height1 = cap1.get(4) # Frame Height
        print('Default width: ' + str(width1) + ', height: ' + str(height1))
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if ret1 == False or ret2 == False:
                print("No camera")
            else:
                timestamp = time.time()
                filename1 = 'frame1' + '.jpg'
                filename2 = 'frame2' + '.jpg'
                cv2.imwrite(filename1,frame1)
                cv2.imwrite(filename2,frame2)
                stitching=cv2.Stitcher.create()
                status,stitched_frame=stitching.stitch((filename1,filename2))

                if status==0:
                    cv2.imshow('stitched_frame',stitched_frame)
                else:
                    print("Error")

            # Display the resulting frame
                side_by_side = cv2.hconcat([frame, frame1])
                cv2.imshow('frame', side_by_side)
                # key: 'ESC'
            key = cv2.waitKey(20)
            if key == 27:
                break
        cap1.release()
        cap2.release() 
        cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()
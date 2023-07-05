import cv2
import time
import platform
def open_camera(index):
    cap = cv2.VideoCapture(index)
    return cap

def main():
    # print OpenCV version
    print("OpenCV version: " + cv2.__version__)
    if platform.system() == 'Windows':
        try:
            import device
            # Get camera list
            device_list = device.getDeviceList()
            index = 0
            for camera in device_list:
                print(str(index) + ': ' + camera[0])
                index += 1
            last_index = index - 1

            if last_index < 0:
                print("No device is connected")
        except:
            print("Not in Windows")
    # Select a camera
    # camera_number = select_camera(last_index)
    # Open camera
    cap1 = open_camerwa(0)
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
                break
            else:
                timestamp = time.time()
                filename1 = 'frame1' + str(timestamp) + '.jpg'
                filename2 = 'frame2' + str(timestamp) + '.jpg'
                cv2.imwrite(filename1,frame1)
                cv2.imwrite(filename2,frame2)
                stitching=cv2.Stitcher.create()
                status,stitched_frame=stitching.stitch((frame1,frame2))

                if status==0:
                    cv2.imshow('stitched_frame',stitched_frame)
                    time.sleep(50) # 50ms
                else:
                    print("Error")
                # put a delay of 100ms between every frame
               

            # Display the resulting frame
                if cap1.isOpened() and cap2.isOpened():
                    side_by_side = cv2.hconcat([frame1, frame2])
                    cv2.imshow('frame', side_by_side)
                else:
                    cv2.imshow('frame', frame1)
                time.sleep(100) # 100ms
                # key: 'ESC'
            key = cv2.waitKey(20)
            if key == 27:
                break
        cap1.release()
        cap2.release() 
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
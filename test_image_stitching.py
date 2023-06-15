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
    cap2= open_camera(1)
    print("1st webcam opened" if cap1.isOpened() else "1st webcam failed to open")
    print("2nd webcam opened" if cap2.isOpened() else "2nd webcam failed to open")

    if cap1.isOpened() or cap2.isOpened():
        width1 = cap1.get(3) # Frame Width
        height1 = cap1.get(4) # Frame Height
        print('Default width: ' + str(width1) + ', height: ' + str(height1))
        # cap1.set(cv2.CAP_PROP_FRAME_WIDTH(640))
        # cap2.set(cv2.CAP_PROP_FRAME_HEIGHT(480))

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if ret1 == False or ret2 == False:
                print("No camera")
            else:
                
                # cv2.save('frame.jpg',frame)
                # cv2.save('frame1.jpg',frame1)
                stitching=cv2.Stitcher.create()
                status,stitched=stitching.stitch((frame1,frame2))
                if status==0:
                    print("Success stitching at" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    cv2.imshow('stitched',stitched)
                else:
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print("Error at " + current_time)
            # Display the resulting frame
                side_by_side = cv2.hconcat([frame1, frame2])
                cv2.imshow('side_by_side', side_by_side)

                # key: 'ESC'
                key = cv2.waitKey(20)
                if key == 27:
                    break

        cap.release()
        cap1.release() 
        cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()
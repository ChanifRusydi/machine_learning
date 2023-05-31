
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
            side_by_side = cv2.hconcat([frame, frame1])
            cv2.imshow('frame', side_by_side)

            # key: 'ESC'
            key = cv2.waitKey(20)
            if key == 27:
                break

        cap.release()
        cap1.release() 
        cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()
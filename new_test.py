import cv2

camera1= cv2.VideoCapture(0)
camera2 = cv2.VideoCapture(1)
camera1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera1.set(cv2.CAP_PROP_FPS, 30)
camera2.set(cv2.CAP_PROP_FPS, 30)
# camera1_placeholder, camera2_placeholder = st.columns([0.5, 0.5])
counter = 0
while(True):
    status1 = camera1.grab()
    
    status2 = camera2.grab()
    print(status1,status2)
    # counter+=1
    
    _,frame1 = camera1.retrieve()
    _,frame2 = camera2.retrieve()
    side_by_side = cv2.hconcat([frame1, frame2])
    cv2.imshow('side by side', side_by_side)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # break
    # if not status1:
    #     print("frame1 is None")
    # elif status1 or status2:
    #     print("frame2 is None")
    #     _,frame1 = camera1.retrieve()
    #     print(type(frame1))
    #     # frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    #     cv2.imshow('frame1', frame1)

camera1.release()
camera2.release()
cv2.destroyAllWindows()
# import the opencv library
# import cv2


# # define a video capture object
# vid = cv2.VideoCapture(0)

# while(True):
	
# 	# Capture the video frame
# 	# by frame
# 	# ret, frame = vid.read()
#     ret = vid.grab()
#     _, frame = vid.retrieve()

# 	# Display the resulting frame
#     cv2.imshow('frame', frame)
	
# 	# the 'q' button is set as the
# 	# quitting button you may use any
# 	# desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # After the loop release the cap object
# vid.release()
# # Destroy all the windows
# cv2.destroyAllWindows()

import cv2

camera1 = cv2.VideoCapture('/dev/video2')
camera2 = cv2.VideoCapture('/dev/video4')

camera1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera1.set(cv2.CAP_PROP_FPS, 15)

camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera2.set(cv2.CAP_PROP_FPS, 15)

print(camera1.isOpened())
print(camera2.isOpened())

while True:
    # 从两个摄像头读取帧
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()

    if ret1:
        cv2.imshow("Camera 1 ", frame1)
    else:
        print("无法从 Camera 2 读取帧")
    
    if ret2:
        cv2.imshow("Camera 2 ", frame2)
    else:
        print("无法从 Camera 2 读取帧")
        
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
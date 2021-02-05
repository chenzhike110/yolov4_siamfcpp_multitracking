import cv2

video = cv2.VideoCapture("rtmp://192.168.43.109:9999/live/test")
while True:
    _, frame = video.read()
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
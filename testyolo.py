from cv2 import cv2
import imutils
import numpy as np
from PIL import Image
from yolo import YOLO

vs = cv2.VideoCapture("./video/video5.mp4")
yolo1 = YOLO()
while True:
    _, frame = vs.read()
    print(frame.shape[1], frame.shape[0])
    frame = frame[:,240:-240]
    cv2.namedWindow('Frame')
    frame = imutils.resize(frame, height=720, width=720)
    frame_clone = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame_clone = Image.fromarray(np.uint8(frame_clone))
    frame = yolo1.detect_image(frame_clone) 
    img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGRA)
    # cv2.imshow("img",img)
    cv2.waitKey(1)
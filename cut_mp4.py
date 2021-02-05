import cv2
 
INPUT_FILE = 't1.avi'
OUTPUT_FILE = 'clip1.avi'
start_frame = 10
end_frame = 50
 
reader = cv2.VideoCapture(INPUT_FILE)
width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(OUTPUT_FILE, 
              cv2.VideoWriter_fourcc("mp4v"),
              30, # fps
              (width, height)) # resolution
 
print(reader.isOpened())
have_more_frame = True
c = 0
while have_more_frame:
    have_more_frame, frame = reader.read()
    c += 1
    if c>= start_frame and c<= end_frame:
        cv2.waitKey(1)
        writer.write(frame)
        print(str(c) + ' is ok')
    if c>end_frame:
        print('completely!')
        break
 
 
writer.release()
reader.release()
cv2.destroyAllWindows()
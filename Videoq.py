import cv2, queue, threading, time

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


if __name__ == '__main__':
  cap = cv2.VideoCapture("/Users/wangyu/Desktop/srtp_code/video_data/video5.mp4")
  cap2=VideoCapture("/Users/wangyu/Desktop/srtp_code/video_data/video5.mp4")
  while True:
    time.sleep(.5)   # simulate time between events
    frame1=cap2.read()
    _,frame = cap.read()
    cv2.imshow("frame", frame)
    cv2.imshow("frame2",frame1)
    if chr(cv2.waitKey(1)&255) == 'q':
      break

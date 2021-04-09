from numba import jit 

class trackthing(object):
    def __init__(self, boxes, clss):
        super().__init__()
        self.boxes = boxes
        self.classes = clss
        self.knn_classes = -1
        self.losted = 0
    def get_boxes(self):
        return self.boxes 
    def get_class(self):
        return self.classes
    def inbox(self, x, y):
        if x>=self.boxes[0] and x<=self.boxes[0]+self.boxes[2] and y>=self.boxes[1] and y<=self.boxes[1]+self.boxes[3]:
            return True
        return False
    def update(self, boxes, clss = None):
        self.boxes = boxes
        self.losted = 0
        if clss != None:
            self.clss = clss
    def knn_update(self, clss):
        self.knn_classes = clss
        self.classes = clss
    def inbox_box(self, rect_box):
        xmin = max(self.boxes[0],rect_box[0])
        ymax = min(self.boxes[1]+self.boxes[3],rect_box[1]+rect_box[3])
        xmax = min(self.boxes[0]+self.boxes[2],rect_box[0]+rect_box[2])
        ymin = max(self.boxes[1],rect_box[1])
        if ymax - ymin < 0 or xmax - xmin < 0:
            return 0
        return (ymax-ymin)*(xmax-xmin)/min(rect_box[2]*rect_box[3], self.boxes[2]*self.boxes[3])
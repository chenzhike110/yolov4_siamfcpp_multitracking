from math import sqrt
import numpy as np
import os
from collections import  Counter


from cv2 import cv2 as cv
import random

import time


modelpath="knn_classes"
online_data_save_path="knn_classes"

class KNNClassifier:

    def __init__(self,modelpath=None):

        self.KNNModel=self.getKNNmodel(modelpath)
    
    def getKNNmodel(self,modelpath):
        KNNModel=[]
        for i in os.listdir(modelpath):
            if i!=".DS_Store":
                for j in os.listdir(os.path.join(modelpath,i)):
                    if j!=".DS_Store":
                        img=cv.imread(os.path.join(modelpath,i,j))
                        m=main_color_moment(img)
                        m.append(str(i))
                        KNNModel.append(m)
        return KNNModel

    def prediction(self,img,k=15):
        m=main_color_moment(img)
        distancelist=[]
        pset=set()
        for i in self.KNNModel:
            d=self.distance(m,i[:-1])
            label=i[-1]
            distancelist.append([d,label])
            pset.add(label)
        
        distancelist=sorted(distancelist)
        pdict={}
        for i in pset:
            pdict[i]=0
        
        for i in range(k):
            pdict[distancelist[i][1]]+=1
        return max(pdict, key=pdict.get)

    def distance(self,p1,p2):
        vector1=np.array(p1)
        vector2=np.array(p2)
        #op1=np.sqrt(np.sum(np.square(vector1-vector2)))
        op2=np.linalg.norm(vector1-vector2)
        return op2


def init_get_video(classname,path=online_data_save_path):
    for i in classname:
        try:
            os.remove(os.path.join(path,i))
        except :
            continue
    for i in classname:
        try:
            os.mkdir(os.path.join(path,i))
        except :
            continue

def get_data_from_video(frame,box,i,classname,path=online_data_save_path):
    """
        从视频中获取前20帧的图像素材
    """
    img=frame[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
    p=os.path.join(path,classname[i%len(classname)],str(i)+".jpg")
    print(p)
    cv.imwrite(p,img)
    #i=i+1

#颜色聚类
def ColorCluster(img):

    #img = cv.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    Z = img.reshape((-1,1))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2 #类别数量
    _,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def main_color_moment(img):
    """
        提取图片的颜色HSV特征
    """
    try:
        img=cv.resize(img,(20,20),interpolation=cv.INTER_CUBIC)
    except Exception as Error:
        print(Error)
        return None
    #img= cv2.GaussianBlur(img,(3,3),0)
    #img= cv2.GaussianBlur(img,(15,15),0)
    # lower_green = np.array([35, 43, 46])
    # upper_green = np.array([78, 255, 255])
    
    # maskgreen = cv2.inRange(img, lower_green, upper_green)
    # masknotgreen = cv2.bitwise_not(maskgreen)
    # img= cv2.bitwise_and(img, img, mask=masknotgreen)
    # N=img.shape[0]*img.shape[1]

    img=cv.cvtColor(img,cv.COLOR_RGB2HSV)
    #img=cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)

    img=ColorCluster(img)
    #m=np.sum(img)/(img.shape[0]*img.shape[1])
    H,S,V=cv.split(img)
    mask1=H-S
    mask2=S-V
    # mask1[mask1!=0]=1
    # mask2[mask2!=0]=1
    mask2[mask1!=0]=1
    img[mask2==0]=0
    H,S,V=cv.split(img)
    muH=np.sum(H)
    H[H!=0]=1
    N=np.sum(H)+1
    muH=muH/N
    muS=np.sum(S)/N
    muV=np.sum(V)/N
    return [muH,muS,muV]#[m,m,m]#
    # H,S,V,N=0,0,0,1

    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i][j][0]==img[i][j][1] and img[i][j][1]==img[i][j][2]:
    #             continue
    #         else:
    #             H=H+img[i][j][0]
    #             S=S+img[i][j][1]
    #             V=V+img[i][j][2]
    #             N=N+1
    # return [H/N,S/N,V/N]


def init_KNN(path=online_data_save_path):
    KNN=KNNClassifier(modelpath=online_data_save_path)
    return KNN


# def judge_by_knn(img,KNN):
#     #m=main_color_moment(img)
#     KNN.prediction(img,KNN)
    


def init_center(path=None):
    centerpoint={}
    for i in os.listdir(path):
        if i!=".DS_Store":
            N,H,S,V=0,0,0,0
            for j in os.listdir(os.path.join(path,i)):
                if j!=".DS_Store":
                    N+=1
                    img=cv.imread(os.path.join(path,i,j))
                    m=main_color_moment(img)
                    #print(m)
                    H=H+m[0]
                    S=S+m[1]
                    V=V+m[2]
                    #print(m)
            centerpoint[i]=[H/N,S/N,V/N]
    return centerpoint

def color_classify(frame,box,centerpoint):
    img=frame[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
    return judge(centerpoint,img)


def judge(centerpoint,img):
    #cv2.imshow("uuu",ColorCluster(img))
    #cv2.waitKey(0)&0xFF
    m=main_color_moment(img)
    #destence={}
    mind=255*255
    keymin=0
    for key,i in centerpoint.items():
        destence=(i[0]-m[0])**2+(i[1]-m[1])**2+(i[2]-m[2])**2
        if mind>destence:
            mind=destence
            keymin=key
    return keymin

def test(test_path,centerpoint):
    fps=[]
    acc={}
    for i in os.listdir(test_path):
        if i!=".DS_Store":
            count=0
            wrong=0
            for j in os.listdir(os.path.join(path,i)):
                if j!=".DS_Store":
                    img=cv.imread(os.path.join(path,i,j))
                    stime=time.time()
                    fact=judge(centerpoint,img)
                    endtime=time.time()
                    fps.append(1/(endtime-stime))
                    if fact!=i:
                        wrong+=1
                    count+=1
            acc[i]=(count-wrong)/count
    print("fps={0:4>.2f}".format(sum(fps)/len(fps)))
    return acc

def get_best_center(centerpoint,data_path):
    #best_center=[]
    err=500
    loss_now=0
    loss_last=0
    count=1
    try_centerpoint=centerpoint
    while loss_now<3:
        loss_now=0
        for key,value in try_centerpoint.items():
            value[0]+=random.choice([-1,1])*random.random()*1/count
            value[1]+=random.choice([-1,1])*random.random()*1/count
            value[2]+=random.choice([-1,1])*random.random()*1/count
            try_centerpoint[key]=value
        for i in os.listdir(data_path):
            if i!=".DS_Store":
                right=0
                num=0
                for j in os.listdir(os.path.join(path,i)):
                    if j!=".DS_Store":
                        img=cv.imread(os.path.join(path,i,j))
                        fact=judge(try_centerpoint,img)
                        if fact==i:
                            right+=1
                        num+=1
                loss_now+=right/num
                #loss_now+=(try_centerpoint[i][0]-m[0])**2+(try_centerpoint[i][1]-m[1])**2+(try_centerpoint[i][2]-m[2])**2
        err=loss_last-loss_now
        print(loss_last,loss_now)
        if err>0:
            try_centerpoint=centerpoint
            err=abs(err)+1
        else:
            loss_last=loss_now
            centerpoint=try_centerpoint
            err=abs(err)
            count+=1
        if count>1000:
            break
        if count%10==0:
            print(centerpoint)
    return centerpoint

# if __name__ == '__main__':
#     centerpoint=init_center(path)
#     print(centerpoint)
#     centerpoint=get_best_center(centerpoint,path)
#     #print(centerpoint)
#     acc=test(test_path,centerpoint)
#     print(acc)
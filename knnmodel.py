import os
import random
import shutil
import time
from collections import Counter
from math import sqrt

import numpy as np
from cv2 import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from STN import STN ,DataSTN,train

modelpath="knn_classes"
online_data_save_path="knn_classes"
classes_name = ["player", "ball", "team1", "team2", "judger"]
class KNNClassifier:

    def __init__(self,modelpath=None):
        epoch=30
        batch_size=32
        device='cpu'
        lr=0.001
        kwargs = {'num_workers': 8, 'pin_memory': True}
        self.transform=transforms.Compose([
                                #transforms.Pad(4),
                                transforms.RandomResizedCrop(28),
                                #transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                            ])

        #self.KNNModel=self.getKNNmodel(modelpath)
        self.SSS=STN()
        self.SSS=torch.load('/home/wxyice/Desktop/srtp/srtp_code/color_data/colorclassification/PPP.pkl')
        for name,p in self.SSS.named_parameters():
            #print(name)
            if name.startswith('localization_linear'): p.requires_grad = False
            if name.startswith('localization_convs'): p.requires_grad = False
            if name.startswith('convs'): p.requires_grad = False
        train_data=DataSTN(modelpath,self.transform)
        train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,**kwargs)
        train(self.SSS,epoch,lr,train_loader,batch_size,device)
        self.SSS.eval()
    
    def getKNNmodel(self,modelpath):
        KNNModel=[]
        for i in os.listdir(modelpath):
            if i!=".DS_Store":
                for j in os.listdir(os.path.join(modelpath,i)):
                    if j!=".DS_Store":
                        img=cv.imread(os.path.join(modelpath,i,j))
                        m=main_color_moment(img)
                        k=[m,str(i)]
                        KNNModel.append(k)
        return KNNModel

    def prediction(self,img,k=5):
        if self.transform:
            img=Image.fromarray(img)
            img=self.transform(img)
            img=img.view(1,3,28,28)
        output = self.SSS(img)
        pred = output.data.max(1, keepdim=True)[1]
        return pred
        # m=main_color_moment(img)
        # distancelist=[]
        # pset=set()
        # for i in self.KNNModel:
        #     d=self.distance(m,i[0])
        #     label=i[-1]
        #     distancelist.append([d,label])
        #     pset.add(label)
        
        # distancelist=sorted(distancelist)
        # pdict={}
        # for i in pset:
        #     pdict[i]=0
        
        # for i in range(k):
        #     pdict[distancelist[i][1]]+=1
        # return classes_name.index(max(pdict, key=pdict.get))

    def distance(self,p1,p2):
        # vector1=np.array(p1)
        # vector2=np.array(p2)
        #op1=np.sqrt(np.sum(np.square(vector1-vector2)))
        try:
            op2=np.linalg.norm(p1-p2)
        except Exception as Error:
            print('[distance wrong]'+Error)
        return op2


def init_get_video(classname,path=online_data_save_path):
    for i in classname:
        try:
            p=os.path.join(path,i)
            shutil.rmtree(p)

        except Exception as Error:
            print(Error)
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
    img=frame[int(box[1]):int(box[1]+box[3]),int(box[0]):int(box[0]+box[2])]
    p=os.path.join(path,classname,str(i)+".jpg")
    print(p)
    try:
        cv.imwrite(p,img)
    except:
        pass
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
def main_color_moment(img)->list:
    try:
        img=cv.resize(img,(30,30),interpolation=cv.INTER_CUBIC)
        img=img[7:23,7:23]
        img=cv.resize(img,(30,30),interpolation=cv.INTER_CUBIC)
    except Exception as Error:
        print(Error)
        return None
    #img= cv.GaussianBlur(img,(3,3),0)
    #img= cv.GaussianBlur(img,(15,15),0)
    # lower_green = np.array([35, 43, 46])
    # upper_green = np.array([78, 255, 255])
    
    # maskgreen = cv2.inRange(img, lower_green, upper_green)
    # masknotgreen = cv2.bitwise_not(maskgreen)
    # img= cv2.bitwise_and(img, img, mask=masknotgreen)
    # N=img.shape[0]*img.shape[1]

    #img=cv.cvtColor(img,cv.COLOR_RGB2HSV)
    img1=cv.cvtColor(img,cv.COLOR_RGB2HSV)
    #img2=cv.cvtColor(img,cv.COLOR_RGBA2GRAY)

    #hist=ColorCluster(img)
    #cv.imshow("F",img)
    #cv.waitKey(0)&0xFF
    #m=np.sum(img)/(img.shape[0]*img.shape[1])
    H,S,V=cv.split(img)
    mask1=H-S
    mask2=S-V
    mask2[mask2!=0]=1
    # mask1[mask1!=0]=1
    # mask2[mask2!=0]=1
    mask2[mask1!=0]=1
    img[mask2==0]=0
    hist1 = cv.calcHist([img], [0], None, [10], [0.0,255.0])
    hist2 = cv.calcHist([img], [1], None, [10], [0.0,255.0])
    hist3 = cv.calcHist([img], [2], None, [10], [0.0,255.0])
    hist4 = cv.calcHist([img1],[0], None, [25], [0.0,255.0])
    hist5 = cv.calcHist([img1],[1], None, [25], [0.0,255.0])
    hist6 = cv.calcHist([img1],[2], None, [25], [0.0,255.0])
    #histg = cv.calcHist([img2],[0], None, [100], [0.0,255.0])
    hist=np.concatenate((hist1,hist2,hist3,hist4,hist5,hist6),axis=0)
    #print(hist.tolist())
    #muH=hist[0]
    #muS=hist[1]
    #muV=hist[2]
    #img[mask2==0]=0
    # H,S,V=cv.split(img)
    # N=img.shape[0]*img.shape[1]
    # muH=np.sum(H)
    # # H[H!=0]=1
    # # N=np.sum(H)+1
    # muH=muH/N
    # muS=np.sum(S)/N
    # muV=np.sum(V)/N
    
    return hist#[muH,muS,muV]#[m,m,m]#
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

# def main_color_moment(img):
#     """
#         提取图片的颜色HSV特征
#     """
#     try:
#         img=cv.resize(img,(20,20),interpolation=cv.INTER_CUBIC)
#     except Exception as Error:
#         print(Error)
#         return None
#     #img= cv2.GaussianBlur(img,(3,3),0)
#     #img= cv2.GaussianBlur(img,(15,15),0)
#     # lower_green = np.array([35, 43, 46])
#     # upper_green = np.array([78, 255, 255])
    
#     # maskgreen = cv2.inRange(img, lower_green, upper_green)
#     # masknotgreen = cv2.bitwise_not(maskgreen)
#     # img= cv2.bitwise_and(img, img, mask=masknotgreen)
#     # N=img.shape[0]*img.shape[1]

#     img=cv.cvtColor(img,cv.COLOR_RGB2HSV)
#     #img=cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)

#     img=ColorCluster(img)
#     #m=np.sum(img)/(img.shape[0]*img.shape[1])
#     H,S,V=cv.split(img)
#     mask1=H-S
#     mask2=S-V
#     # mask1[mask1!=0]=1
#     # mask2[mask2!=0]=1
#     mask2[mask1!=0]=1
#     img[mask2==0]=0
#     H,S,V=cv.split(img)
#     muH=np.sum(H)
#     H[H!=0]=1
#     N=np.sum(H)+1
#     muH=muH/N
#     muS=np.sum(S)/N
#     muV=np.sum(V)/N
#     return [muH,muS,muV]#[m,m,m]#
#     # H,S,V,N=0,0,0,1

#     # for i in range(img.shape[0]):
#     #     for j in range(img.shape[1]):
#     #         if img[i][j][0]==img[i][j][1] and img[i][j][1]==img[i][j][2]:
#     #             continue
#     #         else:
#     #             H=H+img[i][j][0]
#     #             S=S+img[i][j][1]
#     #             V=V+img[i][j][2]
#     #             N=N+1
#     # return [H/N,S/N,V/N]


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
            for j in os.listdir(os.path.join(test_path,i)):
                if j!=".DS_Store":
                    img=cv.imread(os.path.join(test_path,i,j))
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
                for j in os.listdir(os.path.join(data_path,i)):
                    if j!=".DS_Store":
                        img=cv.imread(os.path.join(data_path,i,j))
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

if __name__ == '__main__':
    img=cv.imread('/home/wxyice/Desktop/srtp/srtp_code/color_data/other/C/13.jpg')
    print(img.shape)

    KKK=KNNClassifier()
    print(KKK.prediction(img))

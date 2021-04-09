import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image
import os
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from cv2 import cv2 as cv
from matplotlib import pyplot as plt
import torch.optim as optim
import numpy as np

classes_name = ["player", "ball", "team1", "team2", "judger"]
class DataSTN(Dataset):
    def __init__(self,path,transform):
        self.path=path
        self.classes_name = ["player", "ball", "team1", "team2", "judger"]
        self.transform=transform
        self.classes=[]
        for i in os.listdir(self.path):
            if i!='.DS_Store':
                self.classes.append(str(i))
        self.datapathall=[]
        for classes in self.classes:
            for i in os.listdir(os.path.join(self.path,classes)):
                if i!='.DS_Store':
                    self.datapathall.append([os.path.join(self.path,classes,i),classes])        
        
    def __len__(self):
        return len(self.datapathall)

    def __getitem__(self,index):
        img=cv.imread(self.datapathall[index][0])
        #img=Image.open(self.datapathall[index][0])
        #img = img.convert('RGB')
        #img =img.convert('GRAD')
        img=Image.fromarray(img)
        if self.transform:
            img=self.transform(img)
        label=self.datapathall[index][1]
        label=self.classes_name.index(label)-2
        return img,label

 
class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
      
        self.localization_convs = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.localization_linear = nn.Sequential(
            nn.Linear(in_features=10 * 3 * 3, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2 * 3)
        )
        # 初始化定位网络仿射矩阵的权重/偏置，即是初始化θ值。使得图片的空间转换从原始图像开始。
        self.localization_linear[2].weight.data.zero_()
        self.localization_linear[2].bias.data.copy_(torch.tensor([1, 0, 0,0, 1, 0], dtype=torch.float))
 

        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )

        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=3),
        )
 

    def stn(self, x):
        
        x2 = self.localization_convs(x)
        x2 = x2.view(x2.size()[0], -1)
        x2 = self.localization_linear(x2)
        theta = x2.view(x2.size()[0], 2, 3)  

     
        grid = nn.functional.affine_grid(theta, x.size(), align_corners=True)   # [1, 28, 28, 2]

        x = nn.functional.grid_sample(x, grid, align_corners=True)  # [1, 1, 28, 28]
        return x
 
    def forward(self, x):
        x = self.stn(x)

        x = self.convs(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        x = F.log_softmax(x, dim = 1)
        #print(x)
        return x
        
def train(net,epoch_nums,lr,train_dataloader,per_batch,device):
    net.train()
    #optimizer = optim.Adam(net.parameters(),lr=lr)
    optimizer=optim.Adam(filter(lambda x: x.requires_grad is not False ,net.parameters()), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lossfunc=torch.nn.CrossEntropyLoss()
    #训练模型
    for epoch in range(epoch_nums):
        count=0
        for _,(data,label) in enumerate(train_dataloader):
            #data,label = data.to(device),label.to(device)
            #label=label.ToTensor()
            optimizer.zero_grad()
            pred = net(data)
            #print(pred)
            loss=lossfunc(pred,label)
            loss.backward()
            optimizer.step()
            count+=per_batch
            print("Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}".format(epoch,count,len(train_dataloader.dataset),1. * count /len(train_dataloader),loss.item()))


def test(model,test_loader,device):
    model=torch.load('PPP.pkl')
    model.eval()
    #test_loss = 0
    lossfunc=torch.nn.CrossEntropyLoss()
    correct=0
    for data, target in test_loader:
        
        if device=='cuda':
            data, target = data.cuda(), target.cuda()
        #data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        #loss=lossfunc(output,target)
        # sum up batch loss
        # test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        #print(pred,target,correct)
        #print(loss.item())
    L=len(test_loader.dataset)    
    print(correct/L)

    #test_loss /= len(test_loader.dataset)
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
 
if __name__ == '__main__':

    lr=0.1
    device='cpu'
    transform=transforms.Compose([
                                    #transforms.Pad(4),
                                    transforms.RandomResizedCrop(28),
                                    #transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                ])
    train_path='../other'#'../train'
    test_path='../train'#'../test'

    train_data=DataSTN(train_path,transform)
    test_data=DataSTN(test_path,transform)

    kwargs = {'num_workers': 8, 'pin_memory': True}
    train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,**kwargs)
    test_loader = DataLoader(dataset=test_data,batch_size=1,shuffle=True,**kwargs)
    # for img,label in train_data:
    #     print(img.size(),label)
    SSS=STN()
    SSS=torch.load('PPP.pkl')
    for name,p in SSS.named_parameters():
        #print(name)
        if name.startswith('localization_linear'): p.requires_grad = False
        if name.startswith('localization_convs'): p.requires_grad = False
        if name.startswith('convs'): p.requires_grad = False

    train(SSS,epoch,lr,train_loader,batch_size,device)
    #torch.save(SSS, 'PPP.pkl')
    
    test(SSS,test_loader,device)
        

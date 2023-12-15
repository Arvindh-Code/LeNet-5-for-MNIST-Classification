# -*- coding: utf-8 -*-
'''
Using PyTorch the LeNet5 model has been implemented with 3 different layers in the order of :
Convolution
Avg Pooling
Convolution
Avg Pooling
Convolution
Fully Connected
Fully Connected

And the using model loss through epoch graph will be generate and printing the loss for each epoch approach 
'''
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#Implementing the architecture of LeNet5
def LeNetArchitecture():
  return nn.Sequential(
      #convolution layer-1
      convolutionlayer(1,6,5,1,0),
      #average pooling layer-2
      averagepoolinglayer(2,2),
      #convolution layer-3
      convolutionlayer(6,16,5,1,0),
      #average pooling-4
      averagepoolinglayer(2,2),nn.Flatten(),
      #fully connected layer
      #16 * 4 * 4  = 256
      fullyconnectedlayer(256,120),fullyconnectedlayer(120,84),fullyconnectedlayer(84,10))

#For the Avaerage pooling layer implementation
def averagepoolinglayer(kernel, stride):
        return nn.AvgPool2d(kernel, stride)

#For the convolution layer implementation
def convolutionlayer(in_filter, neurons, kernel, stride, padding):
        return nn.Sequential(nn.Conv2d(in_filter, neurons, kernel, stride, padding),nn.Tanh())

#For the fully connected layer implementation
def  fullyconnectedlayer(in_filter,neurons):
        return nn.Sequential(nn.Linear(in_filter, neurons),nn.Tanh())


#Processing the training data
def training_data(lossArray_train,trainingData,Lenetmodel,lmt):
  for epoch in range(lmt):
    print("Epoch : {}/{}".format(epoch+1,lmt))
    Lenetmodel.train()
    temp_loss=0.0
    for i, j in trainingData:
        optim_.zero_grad()
        loss = criterion(Lenetmodel(i),j)
        loss.backward()
        optim_.step()
        temp_loss=temp_loss+loss.item()
    temp_loss=temp_loss/len(trainingData)
    lossArray_train.append(temp_loss)
    print("---------------------------------------------------------------------")
    print("Epoch : {}, Loss : {}".format(epoch + 1, temp_loss))
    print("---------------------------------------------------------------------")
  return lossArray_train

#Processing the testing data
def testing_data(lossArray_test,siZe,result,Lenetmodel,testData,criterion,lmt):
  Lenetmodel.eval()
  plt.figure(figsize=(100, 60))
  _temp=0.0
  with torch.no_grad():
    for epoch in range(lmt):
      for i, j in testData:
          _,prediction=torch.max(Lenetmodel(i).data,1)
          siZe=siZe+j.size(0)
          _temp=_temp+criterion(Lenetmodel(i),j).item()*i.size(0)
          result=result+(prediction==j).sum().item()
      _temp=_temp/len(testData.dataset)
      lossArray_test.append(_temp)
      for i,(out_,x) in enumerate(testData):
          if i>=1:
            break
          _,prediction=torch.max(Lenetmodel(out_), 1)
          for k in range(out_.size(0)):
            if (k>10):
              break
            plt.subplot(1,out_.size(0),i*out_.size(0)+k+1)
            plt.imshow(out_[k].squeeze())
            plt.title(int(prediction[k]))
  plt.show()
  return lossArray_test,result, siZe

#creating training and testing dataset
trainingData=DataLoader(datasets.MNIST(root='./data',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])),batch_size=64,shuffle=True)
testData=DataLoader(datasets.MNIST(root='./data',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])),batch_size=64,shuffle=False)

# creating a criterion and optimesier for Lenet model
siZe=result=0
lossArray_test=[]
lossArray_train=[]
Lenetmodel = LeNetArchitecture()
crossentropy = nn.CrossEntropyLoss()
optim_ = optim.Adam(Lenetmodel.parameters(), lr=0.001)
#After tried with 100 and 1000 epoch approach the accuracy score were pretty much same with 10 epochs approach. So i went with the 10 epoch approach.
lossArray_train=training_data(lossArray_train,trainingData,Lenetmodel,10) #change the value of epoch if needed
lossArray_test,result,siZe=testing_data(lossArray_test,siZe,result,Lenetmodel,testData,crossentropy,10) #change the value of epoch if needed
accuracy=(result/siZe)*100
print("Test Accuracy : ",accuracy)

plt.plot(lossArray_train)
plt.plot(lossArray_test)
plt.title('Training and Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.show()
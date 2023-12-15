# Master's Projects

   PROGRAMMING ASSIGNMENT 2 REPORT FOR MACHINE LEARNING  

Name
: Aravindh Gopalsamy
UTA ID
: 1002028538
Assignment
: 2
Course
: CSE 6363 - Machine Learning











Implementation of Lenet5 using Pytorch:





As per the architecture Layers have been implemented in the respective function in the code.

LeNetArchitecture(): 

As per the architecture of the lenet5 the layers have been called using their function name and passed with the required Kernal size, stride, padding, and neurons. Following is the order of the layer for the lenet5 implementation Convolution, Average pooling, Convolution, Average pooling, Convolution, and Fully Connected with respective parameters. 

averagepoolinglayer():

Average pooling has been calculated from torch.nn package.

convolutionlayer():

Convolution has been calculated from torch.nn package.

fullyconnectedlayer():

Fully Connected has been calculated from torch.nn package.





training_data():

Data set is passed through the neural network during training. And iterate based on epoch parameters. By using this function, loss between prediction and actual score ,and gradients by backpropagation have been calculated.

testing_data():

After setting the model to eval mode, testing is approached. After the prediction the accuracy score will be shown.

By using this approach able to achieve 98.56% accuracy.



REFERENCE : 

https://github.com/erykml/medium_articles/blob/master/Computer%20Vision/lenet5_pytorch.ipynb


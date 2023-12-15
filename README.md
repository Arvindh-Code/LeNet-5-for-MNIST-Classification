# Implementation of Lenet5 using Pytorch:



![alt text](https://www.google.com/url?sa=i&url=https%3A%2F%2Fhuggingface.co%2Fmindspore-ai%2FLeNet&psig=AOvVaw27z6BeiM_YpGw9JEAozR2M&ust=1702750091345000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCOjdu8aEkoMDFQAAAAAdAAAAABAI)


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


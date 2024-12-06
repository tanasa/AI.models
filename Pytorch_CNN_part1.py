#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://www.simplilearn.com/tutorials/deep-learning-tutorial/convolutional-neural-network
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# https://machinelearningmastery.com/building-a-convolutional-neural-network-in-pytorch/ (possibly for part2)
# https://github.com/MicrosoftDocs/ml-basics/blob/master/05b%20-%20Convolutional%20Neural%20Networks%20(PyTorch).ipynb 


# In[ ]:





# In[ ]:





# In[5]:


print('''
Layers in a Convolutional Neural Network

A convolution neural network has multiple hidden layers that help in extracting information from an image. 
The four important layers in CNN are:

    Convolution layer
    ReLU layer (activation)
    Pooling layer

    Convolution layer
    ReLU layer (activation)
    Pooling layer

    .... 

    Flattening (convert 3D tensor to 1D vector)

    Dense layer
    Output Layer
''')

print(''' 

1. Convolutional Layers :

As the name suggests, the main mathematical task performed is called CONVOLUTION. 
Convolution is the application of a sliding window function (filter, kernel) to a matrix of pixels representing an image. 
This filter detects features like edges or curves in the image.

Consider a 5x5 image whose pixel values are either 0 or 1. A filter matrix may have a dimension of 3x3. 
When we perform edge detection we need to use PADDING.

Slide the filter matrix over the image and compute the dot product to get the convolved feature matrix.
The original image is scanned with multiple convolutions and ReLU layers for locating the features.

2. Pooling Layers

Pooling layers are inserted between convolutional layers.  
Pooling is a down-sampling operation that reduces the dimensionality of the feature map and to control overfitting.

Spatial pooling can be of different types:

    Max Pooling
    Average Pooling
    Sum Pooling

 3. Activation Layer:

The activation layer applies a non-linear activation function, such as the ReLU function, to the output of the pooling layer. 
This function helps to introduce non-linearity into the model, allowing it to learn more complex representations of the input data.

4. Normalization Layer:

The normalization layer performs normalization operations, such as batch normalization or layer normalization, 
to ensure that the activations of each layer are well-conditioned and prevent overfitting.

5. Dropout Layer:

The dropout layer is used to prevent overfitting by randomly dropping out neurons during training. 
This helps to ensure that the model does not memorize the training data but instead generalizes to new, unseen data.

6. Flattening Layer: 

Flattening is used to convert all the resultant 2-Dimensional arrays from pooled feature maps 
into a single long continuous linear vector.

6. Dense Layer:

After the convolutional and pooling layers have extracted features from the input image, 
the dense layer can then be used to combine those features and make a final prediction. 
In a CNN, the dense layer is usually the final layer and is used to produce the output predictions. 

7. Output Layer

To build a CNN, you define the architecture by selecting hyperparameters like number of filters, 
filter size, stride, and pooling size. 

''')


# In[4]:


print(''' 
Types of Convolutional Neural Networks : 

LeNet 

LeNet, developed by Yann LeCun and his team in the late 1990s, is one of the earliest CNN architectures 
designed for handwritten digit recognition. 

It features a straightforward design with two convolutional and pooling layers followed by subsampling, 
and three fully connected layers. Despite its simplicity by today’s standards, LeNet achieved high accuracy 
on the MNIST dataset and laid the groundwork for modern CNNs.


    Parameters: 60k
    Layers flow: Conv -> Pool -> Conv -> Pool -> Flattening -> FC (fully connected, dense) -> FC (fully connected, dense) -> Output
    Activation functions: Sigmoid/tanh and ReLu


AlexNet

AlexNet, created by Alex Krizhevsky and colleagues in 2012, revolutionized image recognition by winning the ImageNet 
Large Scale Visual Recognition Challenge (ILSVRC). Its architecture includes five convolutional layers and 
three fully connected layers, with innovations like ReLU activation and dropout. 
AlexNet demonstrated the power of deep learning, leading to the development of even deeper networks.

ResNet

ResNet, or Residual Networks, introduced the concept of residual connections, allowing the training of very deep networks 
without overfitting.

GoogleNet 

GoogleNet, also known as InceptionNet, introduces the Inception module, which allows the network to process features 
at multiple scales simultaneously. 
''')


# In[ ]:





# In[ ]:


# CIFAR10 datasets


# In[14]:


import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


# In[11]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4


# In[12]:


# 1. Load and normalize CIFAR10

trainset = torchvision.datasets.CIFAR10(root='/home/bogdan/Desktop/PyTorch/CNN', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/home/bogdan/Desktop/PyTorch/CNN', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


# In[17]:


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[18]:


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


# In[19]:


# 2. Define a Convolutional Neural Network

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


# In[20]:


# 3. Define a Loss function and optimizer
# Let’s use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[21]:


# 4. Train the network

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')


# In[22]:


PATH = '/home/bogdan/Desktop/PyTorch/CNN/cifar_net.pth'
torch.save(net.state_dict(), PATH)


# In[23]:


# 5. Test the network on the test data

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


# In[ ]:


# To compute the accuracy


# In[26]:


net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))
outputs = net(images)


# In[27]:


_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))


# In[28]:


# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# In[29]:


# What are the classes that performed well, and the classes that did not perform well:

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


# In[ ]:





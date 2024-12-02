#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://github.com/RileyLazarou/pytorch-generative-models/blob/master/GAN/vanilla_GAN/vanilla_GAN.py
# https://towardsdatascience.com/pytorch-and-gans-a-micro-tutorial-804855817a6b
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/generative_adversarial_network/


# In[2]:


import torch
from torch import nn
import torch.optim as optim

import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os
torch.manual_seed(111)


# In[3]:


# Our GAN script will have three components: 
# a Generator network, 
# a Discriminator network, 
# and the GAN itself, which houses and trains the two networks. 

torch.manual_seed(111)

# Our Generator and Discriminator classes inherit from PyTorch’s nn.Module class, 
# which is the base class for neural network modules.


# In[4]:


# download and use the MNIST datasets
path = '/home/bogdan/Desktop/PyTorch/MNIST'


# In[5]:


device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# In[6]:


# Hyper-parameters
latent_size = 64
hidden_size = 256
image_size = 784

# short run
# num_epochs = 200
num_epochs = 20

batch_size = 100
sample_dir = 'samples'


# In[7]:


# Preparing the Training Data

# The MNIST dataset consists of 28 × 28 pixel grayscale images of handwritten digits from 0 to 9. 
# To use them with PyTorch, you’ll need to perform some conversions. 
# For that, you define transform, a function to be used when loading the data:


# In[8]:


# Define the transform
transform = transforms.Compose([
    transforms.ToTensor(),                # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0.5 and std 0.5
])


# In[9]:


# The function has two parts:

#    transforms.ToTensor() converts the data to a PyTorch tensor.
#    transforms.Normalize() converts the range of the tensor coefficients.

# The original coefficients given by transforms.ToTensor() range from 0 to 1, 
# and since the image backgrounds are black, most of the coefficients are equal to 0 
# when they’re represented using this range.

# The arguments of transforms.Normalize() are two tuples, (M₁, ..., Mₙ) and (S₁, ..., Sₙ), 
# with n representing the number of channels of the images. 
# Grayscale images such as those in MNIST dataset have only one channel, so the tuples have only one value.
# Then, for each channel i of the image, transforms.Normalize() subtracts Mᵢ from the coefficients and 
# divides the result by Sᵢ.


# In[10]:


# MNIST dataset
mnist = torchvision.datasets.MNIST(root='/home/bogdan/Desktop/PyTorch/MNIST/',
                                   train=True,
                                   transform=transform,
                                   download=True)


# In[11]:


# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size, 
                                          shuffle=True)


# In[12]:


# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())


# In[13]:


# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())


# In[14]:


# Device setting
D = D.to(device)
G = G.to(device)


# In[15]:


# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


# In[16]:


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# In[17]:


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# In[18]:


sample_dir = '/home/bogdan/Desktop/PyTorch/MNIST/samples'
os.makedirs(sample_dir, exist_ok=True)


# In[19]:


# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
    
    # Save real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
    
    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))


# In[21]:


# Save the model checkpoints 
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://www.baeldung.com/cs/pytorch-generative-adversarial-networks


# In[2]:


print(''' 
We train two competing and cooperating neural networks called generator (G ) and discriminator or critic (D). 
First, G takes random values from a multivariate Gaussian and produces a synthetic image. After that, 
D learns to distinguish between the real and the generated images. 
The goal of G is to produce a realistic sample that can fool D, whereas D has the opposite goal: 
to learn to differentiate actual from fake images.
''')


# In[3]:


# For image transforms
import torchvision.transforms as transforms
# For Pytorch methods
import torch
import torch.nn as nn
# For Optimizer
import torch.optim as optim
import torchvision
# For DATA SET
from torchvision import datasets, transforms 
# FOR DATA LOADER
from torch.utils.data import DataLoader
# FOR TENSOR BOARD VISUALIZATION
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard


# In[4]:


# Configuration
class Config:
    zDim = 100            # Latent space dimension
    imageDim = 28 * 28    # Flattened size of MNIST images
    batchSize = 64        # Batch size
    # numEpochs = 50      # Number of epochs
    numEpochs = 10  
    lr = 0.0002           # Learning rate
    logStep = 100         # Logging frequency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device


# In[5]:


# Transforms
myTransforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
)


# In[6]:


# MNIST is a handwritten digit database with each digit as a 28X28 monochrome images
# Each image sample is flattened to a 784-dimensional vector (imageDim).


# In[7]:


dataset = datasets.MNIST(root="/home/bogdan/Desktop/PyTorch/MNIST/",
                         transform=myTransforms,
                         download=True)
loader = DataLoader(dataset,
                    batch_size=Config.batchSize,
                    shuffle=True)


# In[8]:


print('''We used a sequential neural network to implement the generator block. 
It comprises an input layer with the Leaky ReLu() activation function, 
followed by a single hidden layer with the tanh() activation function.''')


# In[9]:


# Generator
class Generator(nn.Module):
    def __init__(self, zDim, imgDim, hiddenDim=512, leakySlope=0.01):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(zDim, hiddenDim),
            nn.LeakyReLU(leakySlope),
            nn.Linear(hiddenDim, imgDim),
            nn.Tanh(),  # Output normalized to [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


# In[10]:


print('''The discriminator network is similar to the generator. 
However, its output layer gives a single output 
(1 for real data and 0 for synthetic data)''')


# In[11]:


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, inFeatures, hiddenDim=512, leakySlope=0.01):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(inFeatures, hiddenDim),
            nn.LeakyReLU(leakySlope),
            nn.Linear(hiddenDim, 1),
            nn.Sigmoid(),  # Output probability
        )

    def forward(self, x):
        return self.disc(x)


# In[12]:


# We first create the generator and discriminator objects. 
# Then, we sample the standard Gaussian noise to generate random samples. 
# Following this step, we normalize the monochromatic image map (784-dimensional vector) 
# and convert it to a tensor for processing.

# discriminator = Discriminator(Config.imageDim).to(Config.device)
# generator = Generator(Config.zDim,
#                       Config.imageDim).to(Config.device)

# Fixed Noise
fixedNoise = torch.randn((Config.batchSize,
                              Config.zDim)).to(Config.device)


# In[13]:


# Assuming the output image dimension is 28x28 = 784
imgDim = 28 * 28  # Output image dimension for MNIST dataset
# Model Initialization
gen = Generator(Config.zDim, Config.imageDim).to(Config.device)
disc = Discriminator(Config.imageDim).to(Config.device)


# In[14]:


# Normalization operation reduces each channel’s pixel value with its mean and divides the result with its standard deviation:
# image = image - mu / sigma
# So, transforms.Normalize((0.5,), (0.5,)) converts MNIST image pixel values from the range [0, 1] to [-1, 1]. 
# Hence, this matches the tanh() output of G.


# In[15]:


print(f"\nSetting Optimizers")

# Optimizers
optGen = optim.Adam(gen.parameters(), lr=Config.lr, betas=(0.5, 0.999))
optDisc = optim.Adam(disc.parameters(), lr=Config.lr, betas=(0.5, 0.999))

print(f"Setting the binary cross entropy (BCE) as loss function")

# Loss Function
criterion = nn.BCELoss()


# In[16]:


writerFake = SummaryWriter(f"logs/fake")
writerReal = SummaryWriter(f"logs/real")


# In[17]:


print('''In the training step: we get a random batch of real images from our dataset in each epoch. 
Then, we train our discriminator by showing it synthetic and real images. 
Once that’s over, we train the generator, keeping the discriminator intact. 
Finally, we come to the training step. We get a random batch of real images from our dataset in each epoch. 
Then, we train our discriminator by showing it synthetic and real images. 
Once that’s over, we train the generator, keeping the discriminator intact.
''')


# In[18]:


# Visualization Function
def prepareVisualization(epoch, batch_idx, total_batches, lossD, lossG, writerFake, writerReal, step):
    print(f"Epoch [{epoch}/{Config.numEpochs}], Batch [{batch_idx}/{total_batches}]")
    print(f"Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")

    # Log losses to TensorBoard
    writerFake.add_scalar("Loss/Generator", lossG.item(), step)
    writerReal.add_scalar("Loss/Discriminator", lossD.item(), step)

    # Log fake and real images to TensorBoard
    with torch.no_grad():
        fake = gen(fixedNoise).view(-1, 1, 28, 28)
        real = dataset.data[:16].view(-1, 1, 28, 28).float() / 127.5 - 1
        writerFake.add_images("Generated Images", fake, global_step=step)
        writerReal.add_images("Real Images", real, global_step=step)
    
    step += 1
    return step


# In[19]:


# Training Loop
step = 0
print("\nStarted Training and Visualization...")
for epoch in range(Config.numEpochs):
    print('-' * 80)
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, Config.imageDim).to(Config.device)
        batchSize = real.size(0)
        
        # Train Discriminator
        noise = torch.randn(batchSize, Config.zDim).to(Config.device)
        fake = gen(noise)
        discReal = disc(real).view(-1)
        discFake = disc(fake.detach()).view(-1)
        
        lossDreal = criterion(discReal, torch.ones_like(discReal))
        lossDfake = criterion(discFake, torch.zeros_like(discFake))
        lossD = (lossDreal + lossDfake) / 2

        disc.zero_grad()
        lossD.backward()
        optDisc.step()

        # Train Generator
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        lossG.backward()
        optGen.step()

        # Visualization and Logging
        if batch_idx % Config.logStep == 0:
            step = prepareVisualization(epoch, batch_idx, len(loader), lossD, lossG, writerFake, writerReal, step)



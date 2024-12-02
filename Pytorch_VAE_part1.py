#!/usr/bin/env python
# coding: utf-8

# In[1]:


print(''' 
Autoencoders are a class of generative models. 
They allow us to compress a large input feature space to a much smaller one which can later be reconstructed.
Compression, in general, has got a lot of significance with the quality of learning.

We, humans, have amazing compression capabilities — we are able to learn neat things and later we can easily 
broaden them up when needed. For example, oftentimes, you don’t need to actually remember all the nitty-gritty 
of a particular concept; you just remember specific points about it and later you try to reconstruct it with 
the help of those particular points.

So, if we are able to represent high-dimensional data in a much lower-dimensional space and reconstruct it later, 
it can be very useful for a number of different scenarios like data compression, low-dimensional feature extraction, 
and so on.
''')


# In[2]:


# https://www.linkedin.com/pulse/variational-autoencoder-vae-pytorch-tutorial-shanza-khan-61jyf/
# https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
# https://pyimagesearch.com/tag/autoencoders/ (keras)


# In[3]:


print('''
In traditional autoencoders, inputs are mapped deterministically to a latent vector z = e(x)
In variational autoencoders, inputs are mapped to a probability distribution over latent vectors, 
and a latent vector is then sampled from that distribution. 

The decoder becomes more robust at decoding latent vectors as a result.

Specifically, instead of mapping the input x to a latent vector z = e(x), we map it instead to 
a mean vector μ(x) and a vector of standard deviations σ(x).

These parametrize a diagonal Gaussian distribution  N(μ,σ), from which we then sample a latent vector z∼N(μ,σ).

This is generally accomplished by replacing the last layer of a traditional autoencoder with two layers, 
each of which output μ(x) and σ(x) [KERAS]. 
An exponential activation is often added to σ(x) to ensure the result is positive.
''')


# In[4]:


print('''
Regularization

Regularization in neural networks is a technique used to prevent overfitting by discouraging the model 
from becoming too complex or overly tuned to the training data. 
Overfitting occurs when a neural network performs well on training data but poorly on unseen data. 
Regularization helps the model generalize better to new data by adding constraints or penalties during training.
''')

print('''
Here are some common regularization techniques:

1. L1 and L2 Regularization

These techniques add a penalty term to the loss function based on the size of the weights.

    L1 Regularization (Lasso):
        Adds the sum of the absolute values of the weights (∣w∣∣w∣) to the loss function.
        Encourages sparsity by driving some weights to zero, effectively selecting fewer features.
        Regularized loss:
        L=Lossoriginal+λ∑∣wi∣
        L=Lossoriginal​+λ∑∣wi​∣

    L2 Regularization (Ridge):
        Adds the sum of the squared weights (w2w2) to the loss function.
        Encourages smaller weight magnitudes, which reduces model complexity.
        Regularized loss:
        L=Lossoriginal+λ∑wi2
        L=Lossoriginal​+λ∑wi2​

2. Dropout

    Dropout randomly "drops" (sets to zero) a fraction of the neurons during each forward and backward pass during training. 
    This prevents the network from relying too heavily on specific neurons, encouraging it to distribute learning across the network.

3. Early Stopping

    Stops the training process once the performance on a validation dataset stops improving. 
    This prevents overtraining the model on the training dataset.

4. Data Augmentation

    Expands the training dataset by creating modified versions of existing data (e.g., rotating, flipping, or scaling images). 
    This helps the network generalize better to unseen data.

5. Batch Normalization

    Normalizes the inputs of each layer to reduce internal covariate shift. 
    This acts as a form of regularization by smoothing the optimization landscape.

6. Weight Constraints

    Restricts the range of the weights during training, such as by applying a maximum norm constraint.

7. Adding Noise

    Introduces noise to either the input data, weights, or activation outputs during training. 
    This forces the network to learn more robust features.

''')


# In[5]:


print(''' 
Normalization in artificial neural networks refers to the process of transforming input data or intermediate features 
so that they follow a specific range, distribution, or scale. 

The goal of normalization is to improve the stability and efficiency of the training process, 
reduce the risk of vanishing or exploding gradients, and often accelerate convergence during optimization.

Types of Normalization in Neural Networks:

1    Input Normalization:
        Ensures the input features have similar scales before being fed into the neural network.
        Techniques:
            Min-Max Scaling: Rescales data to a fixed range, such as [0, 1] or [-1, 1].
            X′=X−Xmin / Xmax−Xmin
            Standardization (Z-Score Normalization): 
            Centers the data around zero with a standard deviation of one.
            X′=X−μσ
            X′=σX−μ​
            Improves training stability by preventing features with larger ranges from dominating updates.

2    Batch Normalization:

        Applies normalization to the outputs of a layer within the network 
        (before or after activation functions).
        Normalizes each feature across a batch of data to have zero mean and unit variance:

        μB and σB2 are the mean and variance of the batch, 
        and ϵ is a small constant for numerical stability.
        
        Includes learnable parameters γ (scale) and β (shift) to preserve representational power.
    
        Benefits:
            Reduces internal covariate shift (the change in layer inputs due to parameter updates).
            Acts as a regularizer, potentially reducing the need for dropout.

3    Layer Normalization:

        Similar to batch normalization but normalizes across the features of a single data sample instead of across a batch.
        Particularly useful for recurrent neural networks (RNNs) and situations where batch sizes are small.

4    Instance Normalization:
 
        Normalizes each feature of an individual sample independently, often used in style transfer and image processing tasks.

5    Group Normalization:

        Divides the features into groups and normalizes them within each group.
        Combines benefits of batch and layer normalization, especially useful when batch sizes are very small.

6    Weight Normalization:
 
        Normalizes the weights of the network rather than the activations, improving gradient flow and training stability.

Why Normalization Matters:

    Improves Training Stability: By ensuring features are on a similar scale, optimization algorithms converge more efficiently.
    Accelerates Convergence: Normalized inputs and intermediate layers reduce the number of iterations needed for training.
    Mitigates Vanishing/Exploding Gradients: Keeps the range of activations manageable, particularly in deep networks.

Normalization is a critical preprocessing and architectural step for successful deep learning models. 
The choice of normalization technique depends on the network architecture, type of data, and problem context.
''')


# In[6]:


print('''
VAEs introduce a probabilistic element into the encoding process. 
Namely, the encoder in a VAE maps the input data to a probability distribution over the latent variables, 
typically modeled as a Gaussian distribution with mean μ and variance σ2.

This approach encodes each input into a distribution rather than a single point, adding a layer of variability and uncertainty.

Architectural differences are visually represented by the deterministic mapping in traditional autoencoders 
versus the probabilistic encoding and sampling in VAEs.

This structural difference highlights how VAEs incorporate regularization through a term known as KL divergence, 
shaping the latent space to be continuous and well-structured. 

The regularization introduced significantly enhances the quality and coherence of the generated samples, 
surpassing the capabilities of traditional autoencoders.
''')


# In[7]:


import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid


# In[8]:


# download the MNIST dataset and make dataloaders:

# create a transform to apply to each datapoint
transform = transforms.Compose([transforms.ToTensor()])

# download the MNIST datasets
path = '/home/bogdan/Desktop/PyTorch/MNIST'
train_dataset = MNIST(path, transform=transform, download=True)
test_dataset  = MNIST(path, transform=transform, download=True)

# create train and test dataloaders
batch_size = 100
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example to verify data loading
for images, labels in test_loader:
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels: {labels}")
    break


# In[9]:


print(train_loader)
print(test_loader)


# In[10]:


# Get 25 sample training images for visualization

dataiter = iter(train_loader)
images, _ = next(dataiter)  # Unpack the images and labels

num_samples = 25
sample_images = images[:num_samples]  # Select the first 25 images

fig = plt.figure(figsize=(5, 5))
grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

# Plot each image in the grid
for ax, im in zip(grid, sample_images):
    ax.imshow(im.squeeze(), cmap='gray')  # Remove the channel dimension
    ax.axis('off')

plt.show()


# In[11]:


print('''
We create a simple VAE which has fully-connected encoders and decoders. The input dimension is 784 which is the flattened dimension 
of MNIST images (28×28). In the encoder, the mean (μ) and variance (σ²) vectors are our variational representation vectors (size=200).
Notice that we multiply the latent variance with the epsilon (ε) parameter for reparameterization before decoding. 
This allows us to perform backpropagation and tackle the node stochasticity.
''')


# In[12]:


class VAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, device=device):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


# In[13]:


model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)


# In[14]:


print('''The loss function in VAE consists of reproduction loss and the Kullback–Leibler (KL) divergence. 
The KL divergence is a metric used to measure the distance between two probability distributions.''') 


# In[15]:


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


# In[16]:


# we can train our model:

def train(model, optimizer, train_loader, epochs, device):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            # Flatten the input and move it to the correct device
            x = x.view(x.size(0), -1).to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            x_hat, mean, log_var = model(x)
            
            # Compute loss
            loss = loss_function(x, x_hat, mean, log_var)
            overall_loss += loss.item()
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
        
        # Average loss per sample
        avg_loss = overall_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}\tAverage Loss: {avg_loss:.4f}")
    
    return overall_loss / epochs


# In[17]:


train(model, optimizer, train_loader, epochs=30, device=device)


# In[18]:


print(''' 
We now know that all we need to generate an image from the latent space is two float values (mean and variance). 
Let’s generate some images from the latent space :
''')

def generate_digit(mean, var):
    """
    Generates and visualizes a digit using the latent space coordinates.

    Args:
        mean (float): Value for the first latent dimension (z[0]).
        var (float): Value for the second latent dimension (z[1]).
    """
    # Ensure device is defined globally or replace with appropriate device
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
    with torch.no_grad():  # Disable gradients for decoding
        x_decoded = model.decode(z_sample)
    digit = x_decoded[0].detach().cpu().numpy().reshape(28, 28)  # Reshape to 28x28
    plt.figure(figsize=(1, 1))  # Set smaller figure size
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.show()

# Call the function for two different latent space points
generate_digit(0.0, 1.0)
generate_digit(1.0, 0.0)


# In[19]:


# To plot the latent space :


# In[20]:


def plot_latent_space(model, device, scale=1.0, n=25, digit_size=28, figsize=15):
    """
    Visualizes the latent space of a Variational Autoencoder (VAE).

    Args:
        model: Trained VAE model.
        device: Torch device (e.g., 'cpu' or 'cuda').
        scale: Range of values for the latent space grid (e.g., -scale to +scale).
        n: Number of points in each dimension of the grid.
        digit_size: Size of each digit (usually 28 for MNIST).
        figsize: Size of the overall figure.
    """
    
    # Initialize an empty grid to hold the generated digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # Generate grid points in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]  # Reverse to match image orientation

    # Decode each point in the latent space grid
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            # Create a latent vector z from the grid point
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            
            # Pass through the decoder
            with torch.no_grad():
                x_decoded = model.decode(z_sample)
            
            # Reshape and add to the grid
            digit = x_decoded[0].detach().cpu().numpy().reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size,
                   j * digit_size : (j + 1) * digit_size] = digit

    # Plot the grid of generated digits
    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization')
    plt.xlabel("Latent dimension z[0]")
    plt.ylabel("Latent dimension z[1]")

    # Add axis ticks for the latent space values
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)

    # Display the image grid
    plt.imshow(figure, cmap="Greys_r")
    plt.colorbar(label="Pixel Intensity")
    plt.show()


# In[21]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plot_latent_space(model, device, scale=1.0, n=25, digit_size=28, figsize=15)


# In[22]:


print('''
Changing the scale of the latent space from [−1.0,1.0] to [−5.0,5.0] will affect the range of latent vectors 
fed into the decoder, which can result in the following outcomes:

1. Greater Diversity in Outputs
The latent space [−5.0,5.0] includes points farther from the mean of the normal distribution (z∼N(0,I)) 
typically used to sample latent vectors in VAEs.
This could lead to more varied and extreme outputs, potentially uncovering regions of the latent space 
that the VAE has not been explicitly trained to reconstruct well.

2. Reduced Quality of Reconstruction
Points far from the typical sampling range (e.g., beyond [−3,3] for a standard normal distribution) 
may correspond to latent vectors that the decoder has rarely encountered during training. 
The model might generate low-quality or nonsensical outputs in these regions because they lie 
outside the data's training distribution.

3. Poorer Interpolation
In the latent space of a well-trained VAE, nearby points should decode into similar outputs (smooth interpolation). 
As you move farther from the origin (0, 0) in [−5.0,5.0], the interpolation might become less meaningful, 
as these regions may not correspond to valid or recognizable data.

4. Visualization of Untrained Regions

If the latent space grid includes very extreme values, you might observe:
Blurred or incomplete images.
Images that no longer resemble the dataset (e.g., distorted digits for MNIST).

This happens because the VAE focuses on encoding and decoding regions close to the training data's 
latent space distribution.
''')


# In[23]:


plot_latent_space(model, device=device, scale=5.0, n=25, digit_size=28, figsize=15)


# In[24]:


print('''
By comparing both plots, you’ll notice:

Near [−1.0,1.0]: Outputs are coherent and resemble the training data.
Beyond [−1.0,1.0]: Outputs might degrade as the decoder explores untrained regions.
''')


# In[25]:


print('''
More concretely, this gives as an algorithm that one can use for a diverse set of purposes, including:

Dimensionality Reduction: VAEs can be used to learn a low-dimensional representation of high-dimensional data. 
This is useful for visualization, data compression, and feature extraction. 
In this way, VAEs are similar to PCA, t-SNE, and UMAP, or autoencoders. 

Another way is to think of VAEs as a tool for identifying the intrinsic dimensionality of the data.

Imputation: VAEs can be used to fill in missing data. This is useful for data preprocessing and data augmentation. 
Image in-painting, de-noising, and super-resolution are all examples of imputation.
Generation: VAEs can be used to generate new data. This is useful for data augmentation, data synthesis, and generative modeling.
Image generation, text generation, and music generation are all examples of generation. 
Additionally, if we are mimicking a physical process, we may also be interested in the learned parameters of the model.
''')


# In[ ]:





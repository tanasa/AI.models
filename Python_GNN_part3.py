#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('''More about GRAPHS : adding nodes, end edges :

GNNs can handle unstructured, non-Euclidean data. A graph exists in non-euclidean space. 
It does not exist in 2D or 3D space, which makes it harder to interpret the data. 
To visualize the structure in 2D space, you must use various dimensionality reduction tools.

Graph Convolutional Networks (GCNs) are similar to traditional CNNs. 
It learns features by inspecting neighboring nodes. 
GNNs aggregate node vectors, pass the result to the dense layer, 
and apply non-linearity using the activation function. 

Convolution in Convolutional Neural Network (CNN) is a simple sliding window method over the whole image 
that multiplies the image pixels with the filter weights. 
Similarly, graph convolution uses information from the neighboring nodes to predict features of a given node xi. 

There are two major types of GCNs: Spatial Convolutional Networks and Spectral Convolutional Networks.

Graph Auto-Encoder Networks learn graph representation using an encoder and attempt to reconstruct input graphs using a decoder. 
The encoder and decoders are joined by a bottleneck layer. 
They are commonly used in link prediction as Auto-Encoders are good at dealing with class balance. 
''')


# In[2]:


import networkx as nx
H = nx.DiGraph()

#adding nodes
H.add_nodes_from([
  (0, {"color": "blue", "size": 250}),
  (1, {"color": "yellow", "size": 400}),
  (2, {"color": "orange", "size": 150}),
   (3, {"color": "red", "size": 600})
])

#adding edges
H.add_edges_from([
  (0, 1),
  (1, 2),
  (1, 0),
  (1, 3),
  (2, 3),
  (3,0)
])

node_colors = nx.get_node_attributes(H, "color").values()
colors = list(node_colors)
node_sizes = nx.get_node_attributes(H, "size").values()
sizes = list(node_sizes)

#Plotting Graph
nx.draw(H, with_labels=True, node_color=colors, node_size=sizes)

#converting to undirected graph
G = H.to_undirected()
nx.draw(G, with_labels=True, node_color=colors, node_size=sizes)


# In[3]:


# Beside PyTorch-Geometric, DGL (Deep Graph Library) is another powerful library designed for deep learning on graphs. 
# It simplifies the process of building and training graph neural networks


# In[4]:


print('''
A note about the Spectral Methods : 

Spectral methods perform graph convolution in the spectral domain. 
Graphs are converted from spatial domain to spectral domain using the concept of discrete Fourier transform. 
As the graph is projected to an orthogonal space, a feature matrix U will be obtained from a spectral decomposition 
of a Laplacian matrix. Hence, U is a matrix comprising eigenvalues of corresponding eigenvectors. 

The graph Fourier transform is obtained by taking a dot product of eigenvalues with a function f 
that maps the graph vertices to some numbers on the real line which can ultimately represent as: 
https://www.v7labs.com/blog/graph-neural-networks-guide
''')


# In[ ]:





# In[ ]:





# In[5]:


import networkx as nx
import os
import torch

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


# In[6]:


# Load the CORA dataset
dataset = Planetoid(root='/home/bogdan/Desktop/PyTorch/cora', name='Cora', transform=NormalizeFeatures())

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.
print(data)

# The Cora dataset has 2708 nodes, 10,556 edges, 1433 features, and 7 classes. 
# The first object has 2708 train, validation, and test masks. 
# We will use these masks to train and evaluate the model. 


# In[7]:


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=16)
print(model)


# In[ ]:





# In[8]:


# Visualizating GNN untrained network :

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(6,6))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)


# In[9]:


# Training GNN
# We will train our model on 100 Epochs using Adam optimization and the Cross-Entropy Loss function. 

model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


# In[10]:


def train():
      model.train()
      optimizer.zero_grad()
      out = model(data.x, data.edge_index)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)
      test_correct = pred[data.test_mask] == data.y[data.test_mask]
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
      return test_acc


for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


# In[11]:


# Model evaluation :
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')


# In[12]:


model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y)


# In[ ]:





# In[13]:


# Graph Attention Network
# A popular strategy for improving the aggregation layer in a GNN is to apply ATTENTION.


# In[14]:


print('''

The Graph Attention Networks uses masked self-attentional layers to address the drawbacks of GCNConv. 
A GAT uses attention weights to define  a weighted sum of the neighbours.
The first GNN model to apply this style of attention was introduced by Veličković et al. in 2018. 

Attention Mechanism: 

The attention mechanism assigns weights αij to each neighbor j∈N(i), 
representing how important node jj's features are to node ii.

Learnable Attention Weights: 

Attention coefficients αij are computed as: αij = softmaxj(eij)
where eij is a learnable attention score for the edge between nodes ii and jj.

In the code below, we have just replaced GCNConv with GATConv with 8 attention heads 
in the first layer and 1 in the second layer.
''')


# In[15]:


from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(dataset.num_features, hidden_channels,heads)
        self.conv2 = GATConv(heads*hidden_channels, dataset.num_classes,heads)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GAT(hidden_channels=8, heads=8)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()
      out = model(data.x, data.edge_index)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      return loss

def test(mask):
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)
      correct = pred[mask] == data.y[mask]
      acc = int(correct.sum()) / int(mask.sum())
      return acc

val_acc_all = []
test_acc_all = []

for epoch in range(1, 101):
    loss = train()
    val_acc = test(data.val_mask)
    test_acc = test(data.test_mask)
    val_acc_all.append(val_acc)
    test_acc_all.append(test_acc)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')


# In[16]:


# Model evaluation

import numpy as np

plt.figure(figsize=(6,4))
plt.plot(np.arange(1, len(val_acc_all) + 1), val_acc_all, label='Validation accuracy', c='blue')
plt.plot(np.arange(1, len(test_acc_all) + 1), test_acc_all, label='Testing accuracy', c='red')
plt.xlabel('Epochs')
plt.ylabel('Accurarcy')
plt.title('GATConv')
plt.legend(loc='lower right', fontsize='x-large')
plt.savefig('gat_loss.png')
plt.show()


# In[17]:


model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y)


# In[ ]:





# In[ ]:





# In[ ]:





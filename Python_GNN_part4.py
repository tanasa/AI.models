#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


print('''
NEURAL MESSAGE PASSING is a crucial concept in Graph Neural Networks because it enables information exchange and aggregation 
among nodes in a graph.

Here is how it works : AGGREGATE + UPDATE.

At each iteration of the GNN, the AGGREGATE function takes as input the set of embeddings of the nodes in u’s graph neighbourhood 
N(u) and generates a message m based on this aggregated neighbourhood information. 

The update function UPDATE then combines the message m with the previous embedding of node u to generate the updated embedding.
''')

# https://arxiv.org/pdf/2010.05234
# https://medium.com/@koki_noda/ultimate-guide-to-graph-neural-networks-1-cora-dataset-37338c04fe6f
# https://www.kaggle.com/code/iogbonna/introduction-to-graph-neural-network-with-pytorch


# In[2]:


# GraphSAGE
print("GraphSAGE")
# https://arxiv.org/pdf/2010.05234


# In[3]:


print('''

GraphSAGE (Graph SAmpling and AggreGatE) had made a significant contribution to the GNN research area.
This approach was introduced by the paper Inductive Representation Learning on Large Graphs in 2017. 

Rather than training individual embeddings for each node, the model learns a function that generates embeddings 
by sampling and aggregating the features of the local neighbourhood of a node.

How does GraphSAGE work?

At each iteration, the model follows two different steps:

    Sample: Instead of using the entire neighbourhood of a given node, the model uniformly samples a fixed-size set of neighbours.
    Aggregate: Nodes aggregate information from their local neighbours.

''')


# In[4]:


print("")


# In[5]:


# Analysis 
import os
import torch
from collections import Counter
import matplotlib.pyplot as plt
from torch_geometric.utils import degree
from sklearn.manifold import TSNE
import numpy as np
np.random.seed(0)
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# Visualization
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 24})


# In[6]:


# GAT (Graph Attention Network)
print("Graph Attention Network")


# In[7]:


print('''
Introduced by Veličković et al. in 2017, self-attention in GATs relies on a simple idea: 
some nodes are more important than others. 
In this context, we talk about self-attention (and not just attention) because inputs are compared to each other.

Are transformers a special case on GNN ?

Sentences are fully-connected word graphs

To make the connection more explicit, consider a sentence as a fully-connected graph, 
where each word is connected to every other word. 
Now, we can use a GNN to build features for each node (word) in the graph (sentence), 
which we can then perform NLP tasks with.

Broadly, this is what Transformers are doing: they are GNNs with multi-head attention as the neighbourhood aggregation function. 
Whereas standard GNNs aggregate features from their local neighbourhood nodes , Transformers for NLP treat the entire sentence 
as the local neighbourhood, aggregating features from each word at each layer.
''')

# https://thegradient.pub/transformers-are-graph-neural-networks/
# https://mlabonne.github.io/blog/posts/2022-03-09-Graph_Attention_Network.html


# In[8]:


# A dataset : CiteSeer


# In[9]:


# Import dataset from PyTorch Geometric
dataset = Planetoid(root=".", name="CiteSeer")
data = dataset[0]

# Print information about the dataset
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')


# In[10]:


# Get the list of degrees for each node
degrees = degree(data.edge_index[0]).numpy()

# Count the number of nodes for each degree
numbers = Counter(degrees)

# Bar plot with very small font size
fig, ax = plt.subplots(figsize=(2, 1))        # Smaller figure size
ax.set_xlabel('Node degree', fontsize=2)      # Very small font size
ax.set_ylabel('Number of nodes', fontsize=2)  # Very small font size
plt.bar(numbers.keys(),
        numbers.values(),
        color='#0A047A')

# Adjust tick parameters for very small font size
ax.tick_params(axis='both', which='major', labelsize=2)

# Display the plot
plt.show()


# In[11]:


from torch_geometric.datasets import Planetoid

# Import dataset from PyTorch Geometric
dataset = Planetoid(root=".", name="CiteSeer")
data = dataset[0]

# Print information about the dataset
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')


# In[ ]:





# In[12]:


print("Comparisons between GAT and GCN :")


# In[13]:


# A dataset : Cora


# In[14]:


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


# In[15]:


import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv


class GCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, dim_in, dim_h, dim_out):
      super().__init__()
      self.gcn1 = GCNConv(dim_in, dim_h)
      self.gcn2 = GCNConv(dim_h, dim_out)
      self.optimizer = torch.optim.Adam(self.parameters(),
                                        lr=0.01,
                                        weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.5, training=self.training)
        h = self.gcn1(h, edge_index).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn2(h, edge_index)
        return h, F.log_softmax(h, dim=1)


class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def train(model, data):
    """Train a GNN model and return the trained model."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = model.optimizer
    epochs = 200

    model.train()
    for epoch in range(epochs+1):
        # Training
        optimizer.zero_grad()
        _, out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

        # Print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc*100:.2f}%')
          
    return model

@torch.no_grad()
def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc


# In[16]:


get_ipython().run_cell_magic('time', '', "# Create GCN model\ngcn = GCN(dataset.num_features, 16, dataset.num_classes)\nprint(gcn)\n\n# Train and test\ntrain(gcn, data)\nacc = test(gcn, data)\nprint(f'\\nGCN test accuracy: {acc*100:.2f}%\\n')\n")


# In[17]:


# In this example, we can see that the GAT outperforms the GCN in terms of accuracy (95 % vs. 67.70), 
# but takes longer to train. It’s a tradeoff that can cause scalability issues when working with large graphs.


# In[18]:


print("Embeddings : before any training: these should be random since they’re produced by randomly initialized weight matrices.")


# In[19]:


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Initialize new untrained model
untrained_gat = GAT(dataset.num_features, 8, dataset.num_classes)

# Get embeddings
h, _ = untrained_gat(data.x, data.edge_index)

# Train TSNE
tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(h.detach())

# Plot TSNE visualization with smaller fonts
plt.figure(figsize=(2, 2))
plt.scatter(tsne[:, 0], tsne[:, 1], s=0.1, c=data.y, cmap='viridis')

# Add colorbar with small font size
cbar = plt.colorbar()
cbar.set_label('Node Labels', fontsize=3)  # Very small font size
cbar.ax.tick_params(labelsize=3)  # Small font size for colorbar ticks

# Adjust axes font size (if axes are displayed)
plt.xlabel('TSNE Dimension 1', fontsize=3)  # Small x-axis label font size
plt.ylabel('TSNE Dimension 2', fontsize=3)  # Small y-axis label font size
plt.xticks(fontsize=3)  # Small font size for x-ticks
plt.yticks(fontsize=3)  # Small font size for y-ticks

plt.show()


# In[20]:


get_ipython().run_cell_magic('time', '', "# Create GAT model\ngat = GAT(dataset.num_features, 8, dataset.num_classes)\nprint(gat)\n\n# Train and test\ntrain(gat, data)\nacc = test(gat, data)\nprint(f'\\nGAT test accuracy: {acc*100:.2f}%\\n')\n")


# In[21]:


print("But do the embeddings produced by our trained model look better?")

# Assuming 'gat' is the trained GAT model
# Get embeddings from the trained model
h, _ = gat(data.x, data.edge_index)

# Train TSNE to reduce embeddings to 2D for visualization
tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(h.detach())

# Plot TSNE visualization with smaller fonts
plt.figure(figsize=(2, 2))
plt.scatter(tsne[:, 0], tsne[:, 1], s=0.01, c=data.y, cmap='viridis')

# Add colorbar with small font size
cbar = plt.colorbar()
cbar.set_label('Node Labels', fontsize=3)  # Very small font size
cbar.ax.tick_params(labelsize=3)  # Small font size for colorbar ticks

# Adjust axes font size (if axes are displayed)
plt.xlabel('TSNE Dimension 1', fontsize=3)  # Small x-axis label font size
plt.ylabel('TSNE Dimension 2', fontsize=3)  # Small y-axis label font size
plt.xticks(fontsize=3)  # Small font size for x-ticks
plt.yticks(fontsize=3)  # Small font size for y-ticks

plt.show()


# In[22]:


import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import degree

print('''Nodes with few neighbors are indeed harder to classify. 
This is due to the nature of GNNs: the more relevant connections you have, 
the more information you can aggregate.''')

# Assuming 'gat' is your trained GAT model and 'accuracy' is a defined function

# Get model's classifications
_, out = gat(data.x, data.edge_index)

# Calculate the degree of each node
degrees = degree(data.edge_index[0]).numpy()

# Store accuracy scores and sample sizes
accuracies = []
sizes = []

# Accuracy for degrees between 0 and 5
for i in range(0, 6):
    mask = np.where(degrees == i)[0]
    accuracies.append(accuracy(out.argmax(dim=1)[mask], data.y[mask]))
    sizes.append(len(mask))

# Accuracy for degrees > 5
mask = np.where(degrees > 5)[0]
accuracies.append(accuracy(out.argmax(dim=1)[mask], data.y[mask]))
sizes.append(len(mask))

# Bar plot with reduced font sizes
fig, ax = plt.subplots(figsize=(1, 1))
ax.set_xlabel('Node degree', fontsize=2)  # Smaller font size for x-axis label
ax.set_ylabel('Accuracy score', fontsize=2)  # Smaller font size for y-axis label
ax.set_facecolor('#EFEEEA')

plt.bar(['0', '1', '2', '3', '4', '5', '>5'],
        accuracies,
        color='#0A047A')

# Add smaller annotations for percentages
for i in range(0, 7):
    plt.text(i, accuracies[i], f'{accuracies[i]*100:.2f}%',
             ha='center', color='#0A047A', fontsize=2)  # Smaller text size

# Add smaller annotations for sample sizes
for i in range(0, 7):
    plt.text(i, accuracies[i] / 2, sizes[i],
             ha='center', color='white', fontsize=2)  # Smaller text size

# Reduce tick label sizes
ax.tick_params(axis='both', which='major', labelsize=2)

plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# The code from : https://towardsdatascience.com/how-to-create-a-graph-neural-network-in-python-61fd9b83b54e
# was re-written with GPT4o.


# In[2]:


# GNNs started getting popular with the introduction of the Graph Convolutional Network (GCN) which borrowed some concepts 
# from the CNNs to the graph world. The main idea from this kind of network, also known as Message-Passing Framework, 
# became the golden standard for many years in the area. 

# The Message-Passing framework states that, for every node in our graph, we will do two things:

#    Aggregate the information from its neighbors
#    Update the node information with the information from its previous layer and its neighbor aggregation


# In[ ]:





# In[3]:


import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing, SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from tqdm import tqdm

from torch_geometric.data import Data
# import pyg_lib
import torch_sparse
from tqdm import tqdm


# In[4]:


print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)


# In[5]:


print(''' 
In the following example, GNN architecture uses the SAGE layers.

The SAGE (GraphSAGE) layers are a type of neural network layer used in Graph Neural Networks (GNNs). 
They were introduced in the paper "Inductive Representation Learning on Large Graphs" by Hamilton et al., 2017. 

The main idea behind GraphSAGE (Graph Sample and Aggregate) is to enable inductive learning on graph-structured data, 
where the model can generalize to unseen nodes.

Key Features of SAGE Layers

    Neighborhood Aggregation:
        Each node aggregates feature information from its neighbors to update its representation.
        The aggregation is parameterized and learned, which differentiates GraphSAGE from earlier approaches like GCN 
        (Graph Convolutional Networks).

    Inductive Learning:
        GraphSAGE can generate embeddings for nodes that were not present during training, making it suitable for dynamic graphs.

    Sampling:
        Instead of using all neighbors, GraphSAGE samples a fixed-size subset of neighbors to make the computation scalable on large graphs.

    Aggregation Functions:
        Several aggregation strategies can be used to combine neighbor features:
            Mean Aggregator: Takes the mean of neighbor features.
            Max Pooling Aggregator: Applies a neural network to each neighbor's features and takes the element-wise maximum.
            LSTM Aggregator: Uses an LSTM to combine neighbor features (order-sensitive).
            Concat Aggregator: Concatenates node features with aggregated neighbor features.

''')


# In[6]:


print('''
We must define the number of in_channels of the network, this will be the number of features in our dataset. 
The out_channels is going to be the number of classes we are trying to predict. 
The hidden channels parameter is a value we can define ourselves that represents the number of hidden units.

We must define the number of in_channels of the network, this will be the number of features in our dataset. 
The out_channels is going to be the number of classes we are trying to predict. 
The hidden channels parameter is a value we can define ourselves that represents the number of hidden units.

We can set the number of layers of the network. 
For each hidden layer, we add a Batch Normalization layer and then we reset the parameters for every layer.

We will use the ogbn-arxiv network in which each node is a Computer Science paper on the arxiv and 
each directed edge represents that a paper cited another. 
The task is to classify each node into a paper class.

''')


# In[7]:


# Define the SAGE model

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
        super(SAGE, self).__init__()
        
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        self.layers_bn = torch.nn.ModuleList()
        
        if n_layers == 1:
            self.layers.append(SAGEConv(in_channels, out_channels, normalize=False))
        elif n_layers == 2:
            self.layers.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))
            self.layers.append(SAGEConv(hidden_channels, out_channels, normalize=False))
        else:
            self.layers.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))
            
            for _ in range(n_layers - 2):
                self.layers.append(SAGEConv(hidden_channels, hidden_channels, normalize=False))
                self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))
            
            self.layers.append(SAGEConv(hidden_channels, out_channels, normalize=False))
        
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        if len(self.layers) > 1:
            looper = self.layers[:-1]
        else:
            looper = self.layers
        
        for i, layer in enumerate(looper):
            x = layer(x, edge_index)
            try:
                x = self.layers_bn[i](x)
            except IndexError:
                pass
            finally:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        
        if len(self.layers) > 1:
            x = self.layers[-1](x, edge_index)
        
        return F.log_softmax(x, dim=-1), torch.var(x)

    def inference(self, data, device):
        self.eval()
        with torch.no_grad():
            x, edge_index = data.x.to(device), data.edge_index.to(device)
            out, var = self.forward(x, edge_index)
            return out.cpu(), var.item()


# In[8]:


# Load dataset
target_dataset = 'ogbn-arxiv'
dataset = PygNodePropPredDataset(name=target_dataset, root='networks')
data = dataset[0]
print(data)


# In[9]:


# Data Object Components :
#
#    x:
#        [169343, 128]: This is the node feature matrix. Each of the 169,343 nodes has a feature vector of size 128.
#
#    node_year:
#        [169343, 1]: This represents an additional feature or attribute for each node, such as the year associated with the node. 
#        Each node has a single scalar value for this attribute.
#
#    y:
#        [169343, 1]: This represents the labels or targets for the nodes. 
#        Each node has a single label value, which is often used for tasks like node classification.


# In[10]:


# We could define two Data Loaders to use during our training. 
# The first one will load only nodes from the training set and the second one will load all nodes on the network.

# Notice that we shuffle the training data loader but not the total loader. 
# Also, the number of neighbors for the training loader is defined as the number per layer of the network.

# train_loader = NeighborLoader(data, input_nodes=train_idx,
#                              shuffle=True, num_workers=os.cpu_count() - 2,
#                              batch_size=1024, num_neighbors=[30] * 2)

# total_loader = NeighborLoader(data, input_nodes=None, num_neighbors=[-1],
#                               batch_size=4096, shuffle=False,
#                               num_workers=os.cpu_count() - 2)


# In[11]:


# Train/Val/Test split

split_idx = dataset.get_idx_split()
train_idx = split_idx['train']
valid_idx = split_idx['valid']
test_idx = split_idx['test']


# In[12]:


# Define the model and parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(data.x.shape[1], 256, dataset.num_classes, n_layers=2)
model.to(device)

epochs = 10  # Number of epochs
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=7)


# In[13]:


# Training loop

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()

    # Forward pass using train_mask
    out, _ = model(data.x.to(device), data.edge_index.to(device))
    loss = F.nll_loss(out[train_idx], data.y[train_idx].squeeze().to(device))
    loss.backward()
    optimizer.step()

    # Evaluate model
    model.eval()
    with torch.no_grad():
        train_acc = (out[train_idx].argmax(dim=1) == data.y[train_idx].squeeze().to(device)).float().mean().item()
        val_acc = (out[valid_idx].argmax(dim=1) == data.y[valid_idx].squeeze().to(device)).float().mean().item()
        test_acc = (out[test_idx].argmax(dim=1) == data.y[test_idx].squeeze().to(device)).float().mean().item()

    # Update learning rate scheduler
    scheduler.step(val_acc)

    print(f"Epoch {epoch:02d}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")


# In[ ]:





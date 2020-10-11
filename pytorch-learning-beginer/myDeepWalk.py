#reference: https://dsgiitr.com/blogs/deepwalk/
import torch
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy

adj_list = [[1,2,3], [0,2,3], [0, 1, 3], [0, 1, 2], [5, 6], [4,6], [4, 5], [1, 3]]
size_vertex = len(adj_list)  # number of vertices

# Hyperparameters
w=3            # window size
d=2            # embedding size
y=200          # walks per vertex
t=6            # walk length 
lr=0.025       # learning rate

v=[0,1,2,3,4,5,6,7] # labels of available vertices

def RandomWalk(node,t):
    walk = [node]        # Walk starts from this node
    
    for i in range(t-1):
        node = adj_list[node][random.randint(0,len(adj_list[node])-1)]
        walk.append(node)

    return walk

class DeepWalk(nn.Module):
    def __init__(self):
        super(DeepWalk,self).__init__()
        self.feature_layer = nn.Linear(in_features=size_vertex,out_features=d)
        self.out = nn.Linear(in_features=d,out_features=size_vertex)
    
    def forward(self,one_hot_vector):
        feature_representation = self.feature_layer(one_hot_vector)
        out = self.out(feature_representation)
        return out


myDeepWalk = DeepWalk()

print(myDeepWalk.feature_layer.weight)   #y = x * A.T + b

def skip_gram(wv1,w):
    # wv1: a random walk path
    # w: window size
    optimizer = optim.SGD(myDeepWalk.parameters(),lr=lr)
    for i in range(len(wv1)):
        for j in range(max(0,i-w),min(i+w,len(wv1))):
        
            # generate one-hot vector
            one_hot_vector = torch.zeros(size_vertex)
            one_hot_vector[wv1[i]] = 1
            out = myDeepWalk(one_hot_vector)
            loss = torch.log(torch.sum(torch.exp(out))) - out[wvi[j]]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()       

#deep walk algorithm

for i in range(y):
    random.shuffle(v)
    for vi in v:
        wvi = RandomWalk(vi,t)
        skip_gram(wvi,w)

feature_vector = myDeepWalk.feature_layer.weight.T.detach().numpy()
print(feature_vector)



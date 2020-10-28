import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,dataset
from torchvision import datasets,transforms

#########################################
# References:
# https://github.com/Jiajun-Liu/torchplay/
# https://deeplizard.com/
##########################################

BATCH_SIZE=100
LR=1e-3

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5)
        self.fc = nn.Linear(in_features=120,out_features=84)
        self.out = nn.Linear(in_features=84,out_features=10)
        
    
    def forward(self,input_data):
        af = nn.Tanh()
        #layer 1:conv layer, 6 kernels, size:5*5, stride=1, activation function:tanh()
        layer1_output = af(self.conv1(input_data))
        
        #layer2: subsampling layer, 6 kernels, size:2*2, stride=2
        layer2_input = layer1_output
        layer2_output = F.avg_pool2d(layer2_input,kernel_size=2,stride=2)
        
        #layer3: conv layer, 16 kernels, size=5*5, stride=1, activation function:tanh()
        layer3_input = layer2_output
        layer3_output = af(self.conv2(layer3_input))
        
        #layer4: subsampling layer, 16 kernels, size:2*2, stride=2
        layer4_input = layer3_output
        layer4_output = F.avg_pool2d(layer4_input,kernel_size=2,stride=2)
        
        #layer5: conv layer, 120 kernels, size=5*5, stride=1, activation function:tanh()
        layer5_input = layer4_output
        layer5_output = af(self.conv3(layer5_input))
        
        #layer6: dense layer, in_feature=120, out_feature=84, activation function:tanh()
        layer6_input = layer5_output.reshape(-1,120)
        layer6_output = af(self.fc(layer6_input))
        
        #layer7: output layer, in_feature=84, out_feature=10
        layer7_input = layer6_output
        output = self.out(layer7_input)
        return output

train_set = datasets.MNIST(
    root='./data/MNIST/',
    train=True,
    download=True,
    transform=transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
)
train_loader = DataLoader(train_set,batch_size=BATCH_SIZE)
test_set = datasets.MNIST(
    root='./data/MNIST/',
    train=False,
    download=True,
    transform=transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
)
test_loader =DataLoader(test_set,batch_size=BATCH_SIZE)

def get_correct_nums(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


myLenet = LeNet5()
optimizer = optim.Adam(myLenet.parameters(),lr=LR)

for epoch in range(20):
    total_loss=0
    total_correct=0
    for batch in train_loader:
        images,labels=batch
        preds=myLenet(images)
        loss = F.cross_entropy(preds,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_correct += get_correct_nums(preds,labels)
    
    print("epoch:",epoch,"accuracy:",total_correct/len(train_set),"avg_loss:",total_loss/len(train_set))
    
with torch.no_grad(): 
    test_loss=0
    test_correct=0
    for batch in test_loader:
        images,labels=batch
        preds=myLenet(images)
        loss = F.cross_entropy(preds,labels)
        test_loss += loss.item()
        test_correct += get_correct_nums(preds,labels)
    print("test accuracy: ",test_correct/(len(test_set)), "avg_test_loss:",test_loss/len(test_set))


# final results: train acc: 99.8%  test acc: 98.71%

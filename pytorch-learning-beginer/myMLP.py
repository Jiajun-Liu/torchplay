import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import torchvision
import torchvision.transforms as transforms

# Rerences:
# Thank you for my tutor Jiajun Liu
# https://github.com/Jiajun-Liu/torchplay/blob/master/simplemlp.py
# https://deeplizard.com/
######################################################


class MLPWithThreeHiddenLayers(nn.Module):
    def __init__(self,D_in,H1,H2,H3,D_out):
        super(MLPWithThreeHiddenLayers,self).__init__()
        self.hidden1 = nn.Linear(in_features=D_in, out_features=H1)
        self.hidden2 = nn.Linear(in_features=H1, out_features=H2)
        self.hidden3 = nn.Linear(in_features=H2, out_features=H3)
        self.out = nn.Linear(in_features=H3, out_features=D_out)
        
    def forward(self, t):
        #layer 1: input layer
        t = t
        
        #layer 2: hidden layer 1
        t = self.hidden1(t)
        t = F.relu(t)
        
        #layer 3: hidden layer 2
        t = self.hidden2(t)
        t = F.relu(t)
        
        #layer 4: hidden layer 3
        t = self.hidden3(t)
        t = F.relu(t)
        
        #layer 5: output layer
        t = self.out(t)
        
        return t


def single_batch_without_DataLoader(D_in,H1,H2,H3,D_out,train_x,train_y,test_x):
    #the simplest version
    
    myMLP = MLPWithThreeHiddenLayers(D_in,H1,H2,H3,D_out) #model
    criterion = nn.MSELoss(reduce='mean') #loss function
    optimizer = optim.SGD(params=myMLP.parameters(),lr=1e-4,momentum=0.9) #optimizer
    
    for epoch in range(5):
        
        #myMLP calls __CALL__ method, and then __CALL__ calls forward method which is defined above
        y_pred = myMLP(train_x) 
        
        #compute the loss
        loss = criterion(y_pred,train_y)
        print("in epoch {}, the loss is {}".format(epoch+1,loss.item()))
        
        #backprop & update params: params = params - lr * params.grad
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
    y_out = myMLP(test_x)
    print('y_out.shape ',y_out.shape)


def mini_batch_without_DataLoader(D_in,H1,H2,H3,D_out,batch_size,train_x,train_y,test_x):
    
        
    myMLP = MLPWithThreeHiddenLayers(D_in,H1,H2,H3,D_out) #model
    criterion = nn.MSELoss(reduce='mean') #loss function
    optimizer = optim.SGD(params=myMLP.parameters(),lr=1e-4,momentum=0.9) #optimizer
    
    
    for epoch in range(5):
        
        cursor = 0
        total_loss = 0
        data_left = len(train_x)
        while cursor < len(train_x):
            # devide the dataset
            batch_x = train_x[cursor:cursor+min(batch_size,data_left)]
            batch_y = train_y[cursor:cursor+min(batch_size,data_left)]
            cursor += batch_size
            data_left -= batch_size
            
            #train
            y_pred = myMLP(batch_x) 
            loss = criterion(y_pred,batch_y)
            total_loss += loss.item()
            
#             print(cursor,data_left)
            
            #backprop & update params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("in epoch {}, the loss is {}".format(epoch+1,total_loss)) 
    
    y_out = myMLP(test_x)
    print('y_out.shape ',y_out.shape)


def mini_batch_with_DataLoader(D_in,H1,H2,H3,D_out,batch_size,train_x,train_y,test_x):
    
    class myDataSet(torch.utils.data.Dataset):
        def __init__(self,x,y):
            self.x = x
            self.y = y
            
        def __len__(self):
            return len(self.x)
        
        def __getitem__(self,index):
            return (self.x[index],self.y[index])
        
    train_data = myDataSet(train_x,train_y)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
    
    """
    train_data = torch.utils.data.TensorDataset(train_x,train_y)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
    """
    
    myMLP = MLPWithThreeHiddenLayers(D_in,H1,H2,H3,D_out)
    criterion = nn.MSELoss(reduce='mean')
    optimizer = optim.SGD(params=myMLP.parameters(),lr=1e-4,momentum=0.9)
    
    for epoch in range(5):
        total_loss = 0
        for batch in train_loader:
            datas,labels = batch
            preds = myMLP(datas)
            loss = criterion(preds,labels)
            total_loss += loss.item()
            
            #backprop & update params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("in epoch {}, the loss is {}".format(epoch+1,total_loss)) 
    
    y_out = myMLP(test_x)
    print('y_out.shape ',y_out.shape)

def mini_batch_using_FashionMNIST(D_in,H1,H2,H3,D_out,batch_size):
    #complete processing & model accessment
    
    #dataset 
    train_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST/',
        train=True,
        download=True,
        transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]) 
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    
    test_set = torchvision.datasets.FashionMNIST(
        root = './data/FashionMNIST/',
        train=False,
        download=True,
        transform = transforms.Compose(
        [
            transforms.ToTensor()
        ])
    )
    test_loader = torch.utils.data.DataLoader(dataset=test_set,shuffle=False,batch_size=len(test_set))
    
    #model
    myMLP = MLPWithThreeHiddenLayers(D_in,H1,H2,H3,D_out)
    
    #optimizer 
    
    
    def get_correct_nums(preds,labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    
    
    optimizer = optim.SGD(params=myMLP.parameters(),lr=1e-4,momentum=0.9)

    for epoch in range(100):
        total_loss = 0
        correct_num = 0
        for batch in train_loader:
            images,labels = batch
            images = images.reshape(-1,D_in)

            y_preds = myMLP(images)
            correct_num += get_correct_nums(y_preds,labels)
            loss = F.cross_entropy(y_preds,labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("in epoch {}, avg_loss: {}, accuracy: {}"
              .format(epoch+1, total_loss/(len(train_set)), correct_num/(len(train_set))))
     
    with torch.no_grad():
        for test_images,test_labels in test_loader:
           
            test_images = test_images.reshape(-1,D_in)
            test_preds = myMLP(test_images)
            test_correct_nums = get_correct_nums(test_preds,test_labels)
            test_loss = F.cross_entropy(test_preds,test_labels)
            print("in test: avg_test_loss: {}, test_accuracy: {}".
                  format(test_loss/(len(test_set)),test_correct_nums/len(test_set)))

if __name__ == '__main__':
    N, D_in, H1, H2, H3, D_out, batch_size = 2000, 256, 128, 64, 32, 16, 100
        
    # Create random Tensors to hold training inputs and outputs and test inputs
    train_x = torch.randn(N, D_in)
    train_y = torch.randn(N, D_out)

    test_x = torch.randn(int(N/4), D_in)
    
#     single_batch_without_DataLoader(D_in,H1,H2,H3,D_out,train_x,train_y,test_x)
#     mini_batch_with_DataLoader(D_in,H1,H2,H3,D_out,batch_size,train_x,train_y,test_x)
    D_in,H1,H2,H3,D_out,batch_size = 28*28,128,64,32,10,100
    mini_batch_using_FashionMNIST(D_in,H1,H2,H3,D_out,batch_size)


#the final results: train acc: 83.02% test acc: 81.91%



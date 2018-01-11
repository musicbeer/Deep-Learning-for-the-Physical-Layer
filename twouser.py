#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 15:13:08 2018
two-user autoencoder in paper
@author: musicbeer
"""

import torch
from torch import nn
import numpy as np
NUM_EPOCHS =100
BATCH_SIZE = 32
USE_CUDA = False
parm1=4
parm2=4
M = 2**parm2#one-hot coding feature dim
k = np.log2(M)
k = int(k)
n_channel =parm1#compressed feature dim
R = k/n_channel
CHANNEL_SIZE = M
train_num=8000
test_num=50000
class RTN(nn.Module):
    def __init__(self, in_channels, compressed_dim):
        super(RTN, self).__init__()

        self.in_channels = in_channels

        self.encoder1 = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Tanh(),
            nn.Linear(in_channels, compressed_dim),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(compressed_dim, in_channels),
            nn.Tanh(),
            nn.Linear(in_channels, in_channels)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Tanh(),
            nn.Linear(in_channels, compressed_dim),
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(compressed_dim, in_channels),
            nn.Tanh(),
            nn.Linear(in_channels, in_channels)
        )
    def encode_signal1(self, x):
        x1=self.encoder1(x)
        #x1 = (self.in_channels ** 2) * (x1 / x1.norm(dim=-1)[:, None])
        return x1
    def encode_signal2(self, x):
        x1=self.encoder2(x)
        #x2 = (self.in_channels ** 2) * (x1 / x1.norm(dim=-1)[:, None])
        return x1  
    def decode_signal1(self, x):
        return self.decoder1(x)
    def decode_signal2(self, x):
        return self.decoder2(x)
    def mixedAWGN(self, x1,x2,ebno):
        x1 = (self.in_channels ** 0.5) * (x1 / x1.norm(dim=-1)[:, None])
        # bit / channel_use
        communication_rate = R
        # Simulated Gaussian noise.
        noise1 = Variable(torch.randn(*x1.size()) / ((2 * communication_rate * ebno) ** 0.5))

        
        x2 = (self.in_channels ** 0.5) * (x2 / x2.norm(dim=-1)[:, None])
        # Simulated Gaussian noise.
        noise2 = Variable(torch.randn(*x2.size()) / ((2 * communication_rate * ebno) ** 0.5))
        print("############################",ebno)
        
        signal1=x1+noise1+x2
        signal2=x1+x2+noise2
        return signal1,signal2
    def forward(self, x1,x2):
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        # Normalization.
        x1 = (self.in_channels **0.5) * (x1 / x1.norm(dim=-1)[:, None])
        x2 = (self.in_channels **0.5) * (x2 / x2.norm(dim=-1)[:, None])

        # 7dBW to SNR.
        training_signal_noise_ratio =  5.01187

        # bit / channel_use
        communication_rate = R

        # Simulated Gaussian noise.
        noise1 = Variable(torch.randn(*x1.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        noise2 = Variable(torch.randn(*x2.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        signal1=x1+noise1+x2
        signal2=x1+x2+noise2
        
        decode1 = self.decoder1(signal1)
        decode2 = self.decoder2(signal2)

        return decode1,decode2

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump
if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam,RMSprop
    import torch.utils.data as Data
    model = RTN(CHANNEL_SIZE, compressed_dim=n_channel)
    if USE_CUDA: model = model.cuda()
    train_labels1 = (torch.rand(train_num) * CHANNEL_SIZE).long()
    train_data1 = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=train_labels1)
    train_labels2 = (torch.rand(train_num) * CHANNEL_SIZE).long()
    train_data2 = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=train_labels2)
    train_labels= torch.cat((torch.unsqueeze(train_labels1,1), torch.unsqueeze(train_labels2,1)), 1)
    train_data=torch.cat((train_data1, train_data2), 1)
    
    test_labels1 = (torch.rand(test_num) * CHANNEL_SIZE).long()
    test_data1 = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels1)
    test_labels2 = (torch.rand(test_num) * CHANNEL_SIZE).long()
    test_data2 = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels2)
    test_labels= torch.cat((torch.unsqueeze(test_labels1,1), torch.unsqueeze(test_labels2,1)), 1)
    test_data=torch.cat((test_data1, test_data2), 1)
    dataset = Data.TensorDataset(data_tensor =  train_data, target_tensor = train_labels)
    datasettest = Data.TensorDataset(data_tensor =  test_data, target_tensor = test_labels)
    train_loader = Data.DataLoader(dataset = dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    test_loader = Data.DataLoader(dataset =  datasettest, batch_size = test_num, shuffle = True, num_workers = 2)

    optimizer = Adam(model.parameters(),lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    a=0.5
    b=0.5
    for epoch in range(NUM_EPOCHS):
       for step, (x, y) in enumerate(train_loader):
        b_x1 = Variable(x[:,0:CHANNEL_SIZE])   
        b_y1 = Variable(x[:,0:CHANNEL_SIZE])  
        b_label1 = Variable(y[:,0])               
        b_x2 = Variable(x[:,CHANNEL_SIZE:CHANNEL_SIZE*2])   
        b_y2 = Variable(x[:,CHANNEL_SIZE:CHANNEL_SIZE*2])   
        b_label2 = Variable(y[:,1])               
        decoded1,decoded2 = model(b_x1,b_x2)
        loss1 = loss_fn(decoded1, b_label1)      
        loss2 = loss_fn(decoded2, b_label2)      
        loss=loss1*a+loss2*b

        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()  
        a=loss1/(loss1+loss2)
        a=a.data[0]
        b=loss2/(loss2+loss1)                  # apply gradients
        b=b.data[0]
        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f, L1:%.4f,L2: %.4f,a: %.4f, (1-a):%.4f' % (loss.data[0],loss1.data[0],loss2.data[0],a,b))
            
    import numpy as np
    EbNodB_range = list(frange(0,15.5,0.5))
    ber1 = [None]*len(EbNodB_range)     
    ber2 = [None]*len(EbNodB_range)          
    for n in range(0,len(EbNodB_range)):
     EbNo=10.0**(EbNodB_range[n]/10.0)
     for step, (x, y) in enumerate(test_loader):
        b_x1 = Variable(x[:,0:CHANNEL_SIZE])   
        b_y1 = Variable(x[:,0:CHANNEL_SIZE])  
        b_label1 = Variable(y[:,0])              
        b_x2 = Variable(x[:,CHANNEL_SIZE:CHANNEL_SIZE*2])   
        b_y2 = Variable(x[:,CHANNEL_SIZE:CHANNEL_SIZE*2])   
        b_label2 = Variable(y[:,1])
        encoder1=model.encode_signal1(b_x1)
        encoder2=model.encode_signal2(b_x2)
        encoder1,encoder2=model.mixedAWGN(encoder1,encoder2,EbNo)
        decoder1=model.decode_signal1(encoder1)
        decoder2=model.decode_signal2(encoder2)
        pred1=decoder1.data.numpy()
        pred2=decoder2.data.numpy()
        label1=b_label1.data.numpy()
        label2=b_label2.data.numpy()
        pred_output1 = np.argmax(pred1,axis=1)
        pred_output2 = np.argmax(pred2,axis=1)
        no_errors1 = (pred_output1 != label1)
        no_errors2 = (pred_output2 != label2)
        no_errors1 =  no_errors1.astype(int).sum()
        no_errors2 =  no_errors2.astype(int).sum()
        ber1[n] = no_errors1 / test_num
        ber2[n]=no_errors2 / test_num 
        print ('SNR:',EbNodB_range[n],'BER1:',ber1[n],'BER2:',ber2[n])

#    
## ploting ber curve
    import matplotlib.pyplot as plt
    plt.plot(EbNodB_range, ber1, 'bo',label='Autoencoder1(4,4)')
    plt.yscale('log')
    plt.xlabel('SNR Range')
    plt.ylabel('Block Error Rate')
    plt.grid()
    plt.legend(loc='upper right',ncol = 1)
    
    plt.plot(EbNodB_range, ber2, 'bo',label='Autoencoder2(4,4)',color='r')
    plt.yscale('log')
    plt.xlabel('SNR Range')
    plt.ylabel('Block Error Rate')
    plt.grid()
    plt.legend(loc='upper right',ncol = 1)



#            
#            
#    import matplotlib.pyplot as plt
#    test_labels = torch.linspace(0, CHANNEL_SIZE-1, steps=CHANNEL_SIZE).long()
#    test_data = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels)
#    #test_data=torch.cat((test_data, test_data), 1)
#    test_data=Variable(test_data)
#    x=model.encode_signal1(test_data)
#    x = (n_channel**0.5) * (x / x.norm(dim=-1)[:, None])
#    plot_data=x.data.numpy()
#    plt.scatter(plot_data[:,0],plot_data[:,1],color='r')
#    plt.axis((-2.5,2.5,-2.5,2.5))
#    #plt.grid()
#
#    scatter_plot = []
#
#    scatter_plot = np.array(scatter_plot)
#    print (scatter_plot.shape)
#    
#    test_labels = torch.linspace(0, CHANNEL_SIZE-1, steps=CHANNEL_SIZE).long()
#    test_data = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels)
#    #test_data=torch.cat((test_data, test_data), 1)
#    test_data=Variable(test_data)
#    x=model.encode_signal2(test_data)
#    x = (n_channel**0.5) * (x / x.norm(dim=-1)[:, None])
#    plot_data=x.data.numpy()
#    plt.scatter(plot_data[:,0],plot_data[:,1])
#    plt.axis((-2.5,2.5,-2.5,2.5))
#    plt.grid()
#   # plt.show()
#    scatter_plot = []
##
##    scatter_plot = np.array(scatter_plot)
#    plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 09:42:32 2023

@author: compsysbio
"""

from time import time
import random
import matplotlib.pyplot as plt

from functions import *
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
torch.manual_seed(1729)


# %%import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "cm",
})

#%%
def F(x0,x0seq):
    return x0seq+x0-1
r0 = np.array([1,2,3])
c0 = np.array([1,0.8,0.6])
x0 = np.array(np.meshgrid(c0,r0)).T.reshape(-1,2)

#%%
r_train_cyc = np.load('cyc_res_test.npy')
c_train_cyc = np.load('cyc_cap_test.npy')
th_train_cyc = np.load('cyc_th_test.npy')
c_train_cyc = torch.tensor(c_train_cyc)
r_train_cyc = torch.tensor(r_train_cyc)
th_train_cyc = torch.tensor(th_train_cyc)

#%%
max_th_ = torch.min(torch.max(th_train_cyc,1)[0])

th_ = np.arange(0,max_th_,147)

inds = []
for i in range(len(th_train_cyc)):
    ind = []
    for th in th_:
        ind.append(find_nearest(th_train_cyc[i],th)[0])
        
    inds.append((i,ind))
#%%
cyc_train_matrix = np.load('cyc_train_matrix.npy')
c_train_cyc_= [c_train_cyc[inds[i]] for i in range (len(inds))]
c_train_cyc_ = np.array([np.array(c_train_cyc_[i]) for i in range(len(cyc_train_matrix))])
c_train_cyc_ = torch.tensor(c_train_cyc_)

r_train_cyc_= [r_train_cyc[inds[i]] for i in range (len(inds))]
r_train_cyc_ = np.array([np.array(r_train_cyc_[i]) for i in range(len(cyc_train_matrix))])
r_train_cyc_ = torch.tensor(r_train_cyc_)

CR_train_cyc_vec = []
for c0,r0 in x0:
    d = []
    for i in range(len(c_train_cyc_)):
        d.append(np.stack((F(c0,c_train_cyc_[i]),F(r0,r_train_cyc_[i])),1))
    CR_train_cyc_vec.append(d)
CR_train_cyc_vec=torch.tensor(CR_train_cyc_vec)
CR_train_cyc_vec = CR_train_cyc_vec.squeeze()

torch.save(CR_train_cyc_vec ,'CR_train_cyc_vec.pt')
#%%
CR_train_cyc_vec = torch.load('CR_train_cyc_vec.pt')
CR_train_cyc_vec = CR_train_cyc_vec.transpose(2,3)
# %%


#%%
QQ_vec_train = th_

QQ_train_scaled = (torch.tensor(QQ_vec_train)-torch.tensor(QQ_vec_train)
            .min()).float()/((torch.tensor(QQ_vec_train)).max()-
                             (torch.tensor(QQ_vec_train).min()))
QQs = QQ_train_scaled
QQs= QQs.repeat(CR_train_cyc_vec.shape[0],CR_train_cyc_vec.shape[1],1)
#%%
cyc_train_matrix = torch.tensor(cyc_train_matrix)

DOD = cyc_train_matrix[:, 0]
DOD = DOD.unsqueeze(-2)
DOD = torch.round(DOD, decimals=2)

V_av = (cyc_train_matrix[:, 1]-3.3)/(4.2-3.3)
V_av = V_av.unsqueeze(-2)
V_av = torch.round(V_av, decimals=2)
# %%
tau_time = torch.tile(torch.arange(0, 365, 3), (len(cyc_train_matrix), 1))
tau_time = (tau_time-tau_time.min())/(tau_time.max()-tau_time.min())
#%%
def euler(y0,f,h):
    return y0+h*f(y0)

def rk4(y0, f, h):
    y = [y0]
    # h=(tn-t0)/n

    k1 = h * (f(y0))
    k2 = h * (f(y0+k1/2))
    k3 = h * (f(y0+k2/2))
    k4 = h * (f(y0+k3))
    k = (k1 + 2*k2 + 2*k3 + k4)/6
    y0 = y0+k

    return y0
# %%
# Neural Network Module with Euler method incorporated
torch.manual_seed(172)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 10),
            nn.Tanh(),
            nn.Linear(10, 5)
        )

    def forward(self, throughput, DOD, V_av, C_0, R_0, n):
        y = torch.cat((throughput[:, 0].unsqueeze(-2), DOD,
                      V_av, C_0,R_0), -2)
        y = y.transpose(-2, 1)
        y = y.unsqueeze(-2)
        f = self.linear_relu_stack
        for i in range(1, n+1):
            h = (throughput[:, i] - throughput[:, i-1]
                 ).reshape(len(throughput[:, i]), 1, 1)
            y = torch.cat((y, euler(y[:, -1].unsqueeze(1), f, h)), axis=1)
            y[:, i, 0] = throughput[:, i]
            y[:, i, 1:-2] = y[:, 0, 1:-2]
        return y[:, :, -2:]

model_cyc = NeuralNetwork()
model_cyc = model_cyc.double()

torch.nn.init.xavier_uniform_(model_cyc.linear_relu_stack[0].weight);


torch.nn.init.xavier_uniform_(model_cyc.linear_relu_stack[2].weight);

model_cyc.load_state_dict(torch.load('cyclic_aging_ann.pt'))

#%%Some hyperparameters
learning_rate = 1e-3

loss_fn = nn.MSELoss()

def get_batch_vec(batch_size, n_sample_points):
    s = torch.from_numpy(np.random.choice(np.arange(96-batch_size, dtype=np.int64),
                                          n_sample_points, replace=False))
    true_C = CR_train_cyc_vec[:,:,0]
    true_R = CR_train_cyc_vec[:,:,1]
    Q_eval = QQs
    batch_CR0_vec = torch.stack([CR_train_cyc_vec[j,:,0:2,s] 
                              for j in range(true_C.size(0))], dim=0).transpose(1, 3)
    batch_C0_vec = batch_CR0_vec[:,:,0,:]
    batch_R0_vec = batch_CR0_vec[:,:,1,:]

    batch_CR_vec = torch.stack([torch.stack([CR_train_cyc_vec[j,:,0:2,s+i] for i in range(batch_size)],
                            dim=0).transpose(0, 3) for j in range(true_C.size(0))], dim=0)
    batch_QQs = torch.stack([torch.stack(
        [QQs[j][:, s + i] for i in range(batch_size)], dim=0).transpose(0, 2) 
        for j in range(QQs.size(0))], dim=0)
    return batch_C0_vec,batch_R0_vec,batch_QQs, batch_CR_vec
#%%
loss_list = []
def train_loop(DOD, V_av, model, loss_fn, optimizer,
               batch_size, n_sample_points,n_epochs):

    for epoch in range(n_epochs):
        batch_C0_vec,batch_R0_vec,batch_QQs, batch_CR_vec= get_batch_vec(
            batch_size, n_sample_points)
        for i in range(n_sample_points):
            Q = batch_QQs[:,i,:,:]

            Q0, Qn, n = Q[0, 0], Q[0, -1], len(Q[0][0])-1
            
            CR_true = torch.stack([batch_CR_vec[j,i,:,0:2,:] for j in
                    range(batch_CR_vec.shape[0])], dim=0).transpose(2,3)

            # Compute prediction and loss
            pred = torch.stack([model(Q[j], DOD, V_av,
                    batch_C0_vec[j][i].unsqueeze(0),batch_R0_vec[j][i].
                    unsqueeze(0), n) for j in range(batch_CR_vec.shape[0])], dim=0)
            
            loss = loss_fn(pred, CR_true)
            
            # Backpropagation
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if ((i % 1000) == 0):
                loss_list.append(loss.item())
                print(loss.item())
#%%
optimizer = torch.optim.Adam(model_cyc.parameters(), lr=1e-2)
loss_list = []
st = time()
for i in range(1):
    train_loop(DOD, V_av, model_cyc , loss_fn, optimizer, 
               95,1, 1000)
et = time()
dur = (et - st)/60
print(f'dur={dur}')
#%%
loss_array = np.array(loss_list)

plt.plot(loss_array[loss_array < 1e-5])

plt.plot(loss_list,'*')

def l2_rel_error(actual,pred):
    return np.sqrt(((actual-pred)**2).sum()/(actual**2).sum())
#%%
Cap_0_vec = CR_train_cyc_vec[:, :,0,0]
Res_0_vec = CR_train_cyc_vec[:, :,1,0]


model_cyc = model_cyc .double()
ls = 1
colors = ['blue', 'red', 'yellow', 'green', 'cyan', 'pink', 'black', 'olive', 'purple',
        'indigo', 'tab:blue', 'tab:red', 'brown', 'khaki', 'tab:orange', 'tab:brown']
rc=0
j = 0
o = 0
plt.figure(dpi=500)
l = 0
CC = CR_train_cyc_vec[l,:,rc]
QQ = QQ_vec_train
Cap_0 = Cap_0_vec[l].unsqueeze(-2)
Res_0 = Res_0_vec[l].unsqueeze(-2)
m = model_cyc(QQs[l], DOD, V_av, Cap_0, Res_0, len(QQs[l][0])-1).detach().numpy()
print(l2_rel_error(CR_train_cyc_vec[l].transpose(1,2),m))
for i in [0, 5, 10, 15, 20]:

    plt.plot(QQ/4,m[i+o,:,rc], '-', color=colors[j], linewidth=ls)
    plt.plot(QQ/4,CC[i+o], '.-', color=colors[j], linewidth=ls,markersize=5,alpha=0.4)
    # plt.figure()
    
    j += 1
plt.xlabel('$\\tau$' + ' (in days)')
plt.ylabel('Normalized Capacity')
# plt.legend(fontsize='xx-small')
plt.title('ANN Performance on Samples from the Training Set\n' +
          '(Batch Size = 25)');
#%%
torch.save(model_cyc.state_dict(), 'cyclic_aging_ann_.pt')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 09:42:32 2023

@author: tolu_olola
"""

import torch
from torch import nn

torch.manual_seed(1729)
import numpy as np


import matplotlib.pyplot as plt


#%%import matplotlib.pyplot as plt´
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "cm",
})
#%%
def F(x0,x0seq):
    return x0seq+x0-1

r0 = np.array([1,3])
c0 = np.array([1,0.6])
x0 = np.array(np.meshgrid(c0,r0)).T.reshape(-1,2)

r_train_cal = np.load('cal_res_test.npy')
c_train_cal = np.load('cal_cap_test.npy')

c_train_cal_ = torch.tensor(c_train_cal)
r_train_cal_ = torch.tensor(r_train_cal)
#%%
r_train_cal_.shape

#%%
CR_train_cal_vec = []
for c0,r0 in x0:
    d = []
    for i in range(len(c_train_cal_)):
        d.append(np.stack((F(c0,c_train_cal_[i]),F(r0,r_train_cal_[i])),1))
    CR_train_cal_vec.append(d)
CR_train_cal_vec=torch.tensor(CR_train_cal_vec)
CR_train_cal_vec = CR_train_cal_vec.squeeze()
#Cap_actual_vec = CR_train_cal_vec

torch.save(CR_train_cal_vec ,'CR_train_cal_vec.pt')
#torch.save(Cap_actual_vec,'Res_actual_vec.pt')
#%%
CR_train_cal_vec = torch.load('CR_train_cal_vec.pt')
#Cap_actual_vec = torch.load('Cap_actual_vec.pt')
#%%
CR_train_cal_vec = CR_train_cal_vec.transpose(2,3)
#%%
CR_train_cal_vec.shape
#%%
cal_train_matrix = np.load('cal_train_matrix.npy')
cal_train_matrix=torch.tensor(cal_train_matrix)

V_sto = (cal_train_matrix[:,0]-3.3)/(4.2-3.3)

V_sto = V_sto.unsqueeze(-2)
V_sto = torch.round(V_sto,decimals=2)


Temp = (cal_train_matrix[:,1]-0)/(35-0)

Temp = Temp.unsqueeze(-2)
Temp = torch.round(Temp,decimals=2)
#%%
tau_time = torch.tile(torch.arange(0,365,3),(len(cal_train_matrix),1))

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

#%%
#Neural Network Module with RK4 method incorporated

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 10),
            nn.Tanh(),
            nn.Linear(10, 5)
        )

    def forward(self,t,V_sto, Temp,Cap_0,Res_0,n):
        y = torch.cat((t[:,0].unsqueeze(-2),V_sto,Temp,Cap_0,Res_0),-2)
        y = y.transpose(-2,1)
        y = y.unsqueeze(-2)
        f = self.linear_relu_stack
        for i in range(1,n+1):
            h = (t[:, i] - t[:, i-1]
                 ).reshape(len(t[:, i]), 1, 1)
            y=torch.cat((y,euler(y[:,-1].unsqueeze(1),f,h)),axis=1)
            y[:,i,0] = t[:,i]
            y[:,i,1:-2] = y[:,0,1:-2]
        return y[:,:,-2:]

model_cal = NeuralNetwork()
model_cal=model_cal.double()

torch.nn.init.xavier_uniform_(model_cal.linear_relu_stack[0].weight)

torch.nn.init.xavier_uniform_(model_cal.linear_relu_stack[2].weight)

model_cal.load_state_dict(torch.load('cal_aging_ann.pt'))

#%%
def get_batch_vec(batch_size, n_sample_points):
    s = torch.from_numpy(np.random.choice(np.arange(122-batch_size, dtype=np.int64),
                                          n_sample_points, replace=False))
    true_C = CR_train_cal_vec[:,:,0]
    true_R = CR_train_cal_vec[:,:,1]
    tau_eval = tau_time
    batch_CR0_vec = torch.stack([CR_train_cal_vec[j,:,0:2,s] 
                              for j in range(true_C.size(0))], dim=0).transpose(1, 3)
    batch_C0_vec = batch_CR0_vec[:,:,0,:]
    batch_R0_vec = batch_CR0_vec[:,:,1,:]
    
    batch_CR_vec = torch.stack([torch.stack([CR_train_cal_vec[j,:,0:2,s+i] for i in range(batch_size)],
                            dim=0).transpose(0, 3) for j in range(true_C.size(0))], dim=0)
    batch_tau = torch.stack(
        [tau_time[:, s + i] for i in range(batch_size)], dim=0).transpose(0, 2) 
      
    return batch_C0_vec,batch_R0_vec,batch_tau, batch_CR_vec

#%%Some hyperparameters

loss_fn = nn.MSELoss()
loss_list = []
def train_loop(DOD, V_av, model, loss_fn, optimizer,
               batch_size, n_sample_points,n_epochs):

    for epoch in range(n_epochs):
        batch_C0_vec,batch_R0_vec,batch_tau, batch_CR_vec= get_batch_vec(
            batch_size, n_sample_points)
        for i in range(n_sample_points):
            tau = batch_tau[i]

            tau_0, tau_n, n = tau[0], tau[-1], len(tau[0])-1
            
            CR_true = torch.stack([batch_CR_vec[j,i,:,0:2,:] for j in
                    range(batch_CR_vec.shape[0])], dim=0).transpose(2,3)
            
            # Compute prediction and loss
            pred = torch.stack([model(tau, DOD, V_av,
                batch_C0_vec[j][i].unsqueeze(0),batch_R0_vec[j][i].unsqueeze(0), n)
                                for j in range(batch_CR_vec.shape[0])], dim=0)

            loss = loss_fn(pred, CR_true)

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if ((i % 100) == 0):
                loss_list.append(loss.item())
                print(loss.item())
#%%
loss_list=[]
optimizer = torch.optim.Adam(model_cal.parameters(), lr=1e-3)
from time import time
st = time()
for i in range(1):
    train_loop(V_sto,Temp, model_cal, loss_fn,optimizer,5,117,200)
et = time()
dur = (et - st)/60
print(f'dur={dur}')
#l1[-1]
#%%
loss_array = np.array(loss_list)
plt.figure(dpi=300)
plt.plot(loss_list)
plt.plot(loss_array[loss_array < 1e-8])
def l2_rel_error(actual,pred):
    return np.sqrt(((actual-pred)**2).sum()/(actual**2).sum())
#%%
Cap_0_vec = CR_train_cal_vec[:, :,0,0]
Res_0_vec = CR_train_cal_vec[:, :,1,0]


model_cal = model_cal.double()
ls = 1
colors = ['blue', 'red', 'yellow', 'green', 'cyan', 'pink', 'black', 'olive', 'purple',
        'indigo', 'tab:blue', 'tab:red', 'brown', 'khaki', 'tab:orange', 'tab:brown']
rc=0
j = 0
o = 0

plt.figure(dpi=500)
l = 0
CC = CR_train_cal_vec[l,:,rc]

Cap_0 = Cap_0_vec[l].unsqueeze(-2)
Res_0 = Res_0_vec[l].unsqueeze(-2)
m = model_cal(tau_time,V_sto, Temp, Cap_0,Res_0, len(tau_time[0])-1).detach().numpy()
print(l2_rel_error(CR_train_cal_vec[l].transpose(1,2),m))
for i in [0,5,10,15,20]:

    plt.plot(tau_time[0]*363,m[i+o,:,rc], '--', color=colors[j], linewidth=ls)
    plt.plot(tau_time[0]*363,CC[i+o], '-', color=colors[j], linewidth=ls)
    # plt.figure()'
    
    j += 1
plt.xlabel('$\\tau$' + ' (in days)')
plt.ylabel('Normalized Capacity')
# plt.legend(fontsize='xx-small')
plt.title('ANN Performance on Samples from the Training Set\n' +
          '(Batch Size = 25)');

#%%
torch.save(model_cal.state_dict(), 'cal_aging_ann_.pt')

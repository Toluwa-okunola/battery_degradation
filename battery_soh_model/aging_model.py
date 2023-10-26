#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:13:48 2023

@author: tolu
"""
import numpy as np
import torch
from torch import nn
#%%
def rk4(y0, f, h):
    y = [y0]
    k1 = h * (f(y0))
    k2 = h * (f(y0+k1/2))
    k3 = h * (f(y0+k2/2))
    k4 = h * (f(y0+k3))
    k = (k1 + 2*k2 + 2*k3 + k4)/6
    y0 = y0+k

    return y0

def euler(y0,f,h):
    y0 = y0+h*f(y0)
    return y0
#%%

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
model_cyc.load_state_dict(torch.load('cyclic_aging_ann.pt'))

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
model_cal = NeuralNetwork()
model_cal = model_cal.double()
model_cal.load_state_dict(torch.load('cal_aging_ann.pt'))
#%%
def Normalize(time,throughput,DOD,V_av,Temp):
    time_scaled = (time-0)/(363-0)
    throughput_scaled = (throughput-0)/(max_th_- 0)
    DOD_scaled = DOD
    V_av_scaled = (V_av-3.3)/(4.2-3.3)
    Temp_scaled = (Temp-0)/(35-0)
    return time_scaled,throughput_scaled,DOD_scaled,V_av_scaled,Temp_scaled

def aging_model(time,throughput,DOD,V,Temp,Cap_0,Res_0):
    '''
    Parameters
    ----------
    time : an array of timesteps in days.
    throughput : an array of throughput time steps in AmpereHours
    DOD : an array of depths of discharge corresponding to the timesteps. 
                Enter 0 for calendar aging (when the battery is at rest).
    V : an array of average voltages corresponding to the timesteps.
    Temp : an array of measured temperatures corresponding to the timesteps.
    Cap_0 : the initial relative capacity in (divide last measured capacity
                                by nominal internal resistance: 71.6e-3 ohms).
    Res_0 : the initial relative internal resistance (divide last measured 
            internal resistance by nominal internal resistance: 71.6e-3 ohms).
    Returns
    -------
    Res_det : The predicted relative internal restistance at the provided time steps.
    Cap_det : The predicted relative capacity at the provided time steps.

    '''
    def Normalize(time,throughput,DOD,V_av,Temp):
        max_th_ = 14026.7015
        time_scaled = (time-0)/(363-0)
        throughput_scaled = (throughput-0)/(max_th_- 0)
        DOD_scaled = DOD
        V_av_scaled = (V_av-3.3)/(4.2-3.3)
        Temp_scaled = (Temp-0)/(35-0)
        return time_scaled,throughput_scaled,DOD_scaled,V_av_scaled,Temp_scaled
    args = [time,throughput,DOD,V,Temp]
    for i, arg in enumerate(args):
        if (type(arg) != torch.Tensor):
            args[i] = torch.tensor(arg)
    time,throughput,DOD,V,Temp = args
    time,throughput,DOD,V,Temp = Normalize(time,throughput,DOD,V,Temp)
    Res_k = torch.ones(1,1)*Res_0
    Cap_k = torch.ones(1,1)*Cap_0
    Res_vec = [Res_0]
    Cap_vec = [Cap_0]
    for k in range(len(throughput)-1):
        DOD_k = DOD[k].reshape(1,1)
        V_k = V[k].reshape(1,1)
        Temp_k = Temp[k].reshape(1,1)
        Q_k = throughput[k:k+2].unsqueeze(0)
        tau_k = time[k:k+2].unsqueeze(0)

        if DOD_k == 0:
            cal_aging = model_cal(tau_k,V_k,Temp_k,Cap_k,Res_k,1)[:,-1]
            Res_vec.append(cal_aging[:,1].squeeze())
            Cap_vec.append(cal_aging[:,0].squeeze())
        else: 
            cyc_aging = model_cyc(Q_k,DOD_k,V_k,Cap_k,Res_k,1)[:,-1]
            cal_aging = model_cal(tau_k,V_k,Temp_k,Cap_k,Res_k,1)[:,-1]
            Res_vec.append((cyc_aging[:,1] +  
                    cal_aging[:,1] - Res_k).squeeze())
            Cap_vec.append((cyc_aging[:,0] +  
                    cal_aging[:,0] - Cap_k).squeeze())
        Res_k = Res_vec[-1].reshape(1,1)
        Cap_k = Cap_vec[-1].reshape(1,1)
        
    Res_det = [Res_0]
    for i in range(1,len(Res_vec)):
        Res_det.append(Res_vec[i].detach().numpy())
        
    Cap_det = [Cap_0]
    for i in range(1,len(Cap_vec)):
        Cap_det.append(Cap_vec[i].detach().numpy())
        
    return Res_det,Cap_det
#%%
help(aging_model)

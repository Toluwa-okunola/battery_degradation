#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:13:48 2023

@author: compsysbio
"""
#from aging_model_settings import *
from aging_model import *

import matplotlib.pyplot as plt

from kalman_filters import *

from functions import *
import numpy as np
import torch
from torch import nn

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "cm",
})
#%%
def l2_rel_error(actual,pred):
    return np.sqrt(((actual-pred)**2).sum()/(actual**2).sum())
TIMES_DS = [[0, 7, 8.2, 16.5, 17.7, 24],
           [0, 4, 8, 16, 20, 24],
           [0, 6, 12, 18, 24],
           [0, 4, 8, 16, 20, 24],
           [0,24]]

DODS = [[0, 0.2, 0, 0.2, 0],
        [0.6,0,0.4,0,0.3],
        [0,0.5,0,0.3],
        [0.5]*5,
        [0.5]]

VS = [V_oc(np.array([0.87, 0.87, 0.87, 0.87, 0.87])),
      V_oc(np.array([0.5, 0.68, 0.4, 0.6, 0.6])),
      V_oc(np.array([0.5]*4)),
      V_oc(np.array([0.55]*5)),
      V_oc(np.array([0.55]))]

T_ENS = [35]*5

C_RATES = [0.48,0.4,0.3,0.05,0.05]

times_=[]
soc_profiles = []
actual_trajs = []
learned_trajs = []
errors=[]

for i in range(5):
    
    times_d = TIMES_DS[i] 
    Vs = VS[i]
    T_en = T_ENS[i]
    Dods = DODS[i]
    c_rate = C_RATES[i]
    soc_profile,T, V, Is = Use(t, times_d, Vs, T_en, Dods, c_rate, C0*0.7, R0*1.9)
    
    le = np.array(times_d)*3000/24
    le = le.astype(int)
    Ts = [T_en]*len(Dods)
    qs = np.array([0]+[np.trapz(abs(Is[le[i]:le[i+1]]), t[le[i]:le[i+1]])
                       for i in range(len(le)-1)])
    
    days = 365
    qq = np.tile(qs[1:], days).cumsum()
    
    times = []
    for day in range(days):
        times.append(day+np.array(times_d[1:])/24)
    
    times = np.insert(np.array(times).reshape(
        len(Dods)*days), 0, 0)  # +60#+50#+100
    qq = np.insert(qq, 0, 0) 
    
    Dods_ = np.tile(np.array(Dods), days)
    Vs_ = np.tile(np.array(Vs), days)
    Ts_ = np.tile(np.array(Ts), days)
    
    #actual trajectory of resistance and capacity
    Rs, Cs = Cap_and_res_real(times, qq, Dods_, Vs_, Ts_, c0, r0)
    
    #trajectory of resistance and capacity produced by the model
    rs, cs = np.array(aging_model(times, qq, Dods_, Vs_, Ts_, c0, r0))
    
    actual_trajs.append((Rs,Cs))
    learned_trajs.append((rs,cs))
    soc_profiles.append(soc_profile)
    errors.append((l2_rel_error(Rs,rs),l2_rel_error(Cs,cs)))
    times_.append(times)
#%%
plt.figure(dpi=300)
fig, axs = plt.subplots(1, 5, dpi=300, figsize=(15, 3))
for i in range(5):
    axs[i].plot(t, soc_profiles[i], color='tab:blue')
    axs[i].set_xlabel('time'+'(hrs)')
    axs[i].set_xticks([0,8,16,24])
    axs[0].set_ylabel('SOC')
#%%
plt.figure(dpi=300)
fig, axs = plt.subplots(1, 5, dpi=300,figsize=(15, 3))
for i in range(5):
    axs[i].plot(times_[i],actual_trajs[i][1], color='tab:blue',
                label='Normalized True Resistance')
    axs[i].plot(times_[i],learned_trajs[i][1],'--',color='skyblue',
                label='Normalized Learned Resistance')
    axs[i].plot(times_[i], actual_trajs[i][0], color='tab:red',
                label='Normalized True Capaciity')
    axs[i].plot(times_[i],learned_trajs[i][0],'--',color='salmon',
                label='Normalized Learned Capacity')
    axs[i].set_xlabel('$\\tau$'+'(days)')
    axs[0].set_ylabel('Normalized Capacity/Resistance')

    plt.legend(bbox_to_anchor=(-1.3, -0.45), loc='lower center', ncol=2)
#%%
T_n = T + np.random.normal(0,1e-5,T.shape)
V_n = V + np.random.normal(0,1e-2,V.shape)
#%%
plt.plot(T_n)
plt.plot(T)
#%%
res=ekf_Res_single(t,T_n,Is,noise_sd =1e-2)
plt.plot(res)
res[-1],R0*1.9
#%%
plt.plot(V_n)
plt.plot(V)
#%%
cap=ekf_Cap_single(t,V_n,Is,res[-1],noise_sd=5*1e-3)
plt.plot(cap)
#%%
cap[-1],C0*0.7


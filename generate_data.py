#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:40:19 2023

@author: compsysbio
"""
import numpy as np
from functions import *
import matplotlib.pyplot as plt
from kalman_filters import *
#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "cm",
    })
#%%
#Cycling_test_matrix

def cyc_aging_test_matrix_func():
    DOD_test = np.array([0.15,0.25,0.4,0.6,0.75,0.85])
    S_test= np.array([0.25,0.35,0.45,0.55,0.65,0.75])
    
    temp = np.array(np.meshgrid(DOD_test,S_test)).T.reshape(-1,2)
    
    temp = temp[~((temp[:,1]==0.75) * (temp[:,0]>0.5))]
    temp = temp[~((temp[:,1]==0.65) * (temp[:,0]>0.7))]
    temp = temp[~((temp[:,1]==0.55) * (temp[:,0]>0.9))]
    temp = temp[~((temp[:,1]==0.45) * (temp[:,0]>0.9))]
    temp = temp[~((temp[:,1]==0.35) * (temp[:,0]>0.7))]
    temp = temp[~((temp[:,1]==0.25) * (temp[:,0]>0.5))]
    #plt.plot(temp[:,0],temp[:,1],'*')
    temp[:,1] = V_oc(temp[:,1])
    cyc_test_matrix = temp
    plt.figure(dpi = 300)
    plt.plot(cyc_test_matrix[:,0],cyc_test_matrix[:,1],'.',color='tab:green',markersize=10)
    plt.xlabel('Cycle Depth')
    plt.xticks(cyc_test_matrix[:,0])
    plt.yticks(np.round(cyc_test_matrix[:,1],2))
    plt.grid(color='tab:blue', linestyle='--', linewidth=0.5)
    plt.ylabel('Average Voltage')
    plt.savefig('Cyclic Aging Test Matrix.png')
    print(len(cyc_test_matrix))
    np.save('cyc_test_matrix.npy',cyc_test_matrix)
    return cyc_test_matrix
#%%
def cyc_aging_train_matrix_func():
    #DOD_train = np.array([0.1,0.2,0.3,0.5,0.7,0.8,0.9])
    DOD_train = np.array([0.1,0.3,0.5,0.7,0.8,0.9])
    S_train= np.array([20,30,40,50,60,70,80])/100
    
    temp = np.array(np.meshgrid(DOD_train,S_train)).T.reshape(-1,2)
    
    temp = temp[~((temp[:,1]==0.8) * (temp[:,0]>0.4))]
    temp = temp[~((temp[:,1]==0.7) * (temp[:,0]>0.6))]
    temp = temp[~((temp[:,1]==0.6) * (temp[:,0]>0.8))]
    temp = temp[~((temp[:,1]==0.4) * (temp[:,0]>0.8))]
    temp = temp[~((temp[:,1]==0.3) * (temp[:,0]>0.6))]
    temp = temp[~((temp[:,1]==0.2) * (temp[:,0]>0.4))]
    temp = temp[~(((temp[:,1]>=0.4) * (temp[:,1]<=0.6))* (temp[:,0]==0.2))]

    temp[:,1] = V_oc(temp[:,1])
    cyc_train_matrix = temp
    plt.figure(dpi = 300)
    plt.plot(cyc_train_matrix[:,0],cyc_train_matrix[:,1],'.',color='tab:green',markersize=10)
    plt.xlabel('Cycle Depth')
    plt.xticks(cyc_train_matrix[:,0])
    plt.yticks(np.round(cyc_train_matrix[:,1],2))
    plt.grid(color='tab:blue', linestyle='--', linewidth=0.5)
    plt.ylabel('Average Voltage')
    plt.savefig('Cyclic Aging Training Matrix.png')
    print(len(cyc_train_matrix))
    np.save('cyc_train_matrix.npy',cyc_train_matrix)
    return cyc_train_matrix
#%%
def cal_aging_train_matrix_func():
    V_sto = np.array([3.4,3.6,3.75,3.9,4.1])
    T_sto = np.array([10,15,22.5,30,35])

    cal_train_matrix = np.array(np.meshgrid(V_sto,T_sto)).T.reshape(-1,2)
    plt.figure(dpi = 300)
    plt.plot(cal_train_matrix[:,0],cal_train_matrix[:,1],'.',color='tab:green',markersize=10)
    plt.ylabel('Storage Temperature')
    plt.xticks(cal_train_matrix[:,0])
    plt.yticks(cal_train_matrix[:,1])
    plt.grid(color='tab:blue', linestyle='--', linewidth=0.5)
    plt.xlabel('Storage Voltage')
    plt.savefig('Calendar Aging Training Matrix.png')
    print(len(cal_train_matrix))
    np.save('cal_train_matrix.npy',cal_train_matrix)
    return cal_train_matrix
#%%
def cal_aging_test_matrix_func():
    V_sto = np.array([3.5,3.7,3.8,4.0])
    T_sto = np.array([12.5,17.5,22.5,27.5,32.5])

    cal_test_matrix = np.array(np.meshgrid(V_sto,T_sto)).T.reshape(-1,2)
    plt.figure(dpi = 300)
    plt.plot(cal_test_matrix[:,0],cal_test_matrix[:,1],'.',color='tab:green',markersize=10)
    plt.ylabel('Storage Temperature')
    plt.xticks(cal_test_matrix[:,0])
    plt.yticks(cal_test_matrix[:,1])
    plt.grid(color='tab:blue', linestyle='--', linewidth=0.5)
    plt.xlabel('Storage Voltage')
    plt.savefig('Calendar Aging Test Matrix.png')
    print(len(cal_test_matrix))
    np.save('cal_test_matrix.npy',cal_test_matrix)
    return cal_test_matrix
#%%
def gen_cal_cap_res_training_data(Cal_train_matrix,noise_sd_t = 0.0001,noise_sd_v = 0.0001):
    t = np.linspace(0,1,300)
    tau = np.arange(0,365,3)
    C0 = 2.05
    R0 = 71.6e-3
    Volt_meas = []
    Temp_meas = []
    Res_meas = []
    Cap_meas = []
    Volt_noised = []
    Temp_noised = []
    SOC_meas = []
    Current_meas = []

    for V_sto, T_sto in Cal_train_matrix:
        TT_cal_noised = []
        VV_cal_noised = []
        SS_cal, TT_cal, VV_cal, II_cal, CC_cal, RR_cal = Cal_ageing_tests(t,tau,V_sto,T_sto,0.2,C0,R0) 
        for i in range(len(TT_cal)):
            TT_cal_noised.append(TT_cal[i] + np.random.normal(0,noise_sd_t,len(TT_cal[i])))
            VV_cal_noised.append(VV_cal[i] + np.random.normal(0,noise_sd_v,len(VV_cal[i])))
        Volt_meas.append(VV_cal)
        Temp_meas.append(TT_cal)
        Volt_noised.append(VV_cal_noised)
        Temp_noised.append(TT_cal_noised)
        Res_meas.append(RR_cal)
        Cap_meas.append(CC_cal)
        SOC_meas.append(SS_cal)
        Current_meas.append(II_cal)
    res = Kalman_Filter_Res(t,Temp_noised,Current_meas,0.47)*1e3
    cap = Kalman_Filter_Cap(t,Volt_noised,Current_meas,res,0.2)
    #np.save('cal_cap_train.npy',cap/2.05)
    #np.save('cal_res_train.npy',res/71.6e-3)
    #np.save('cal_cap_train_true.npy',np.array(Cap_meas)/2.05)
    #np.save('cal_res_train_true.npy',np.array(Res_meas)/71.6e-3)
    return res, cap,np.array(Res_meas),np.array(Cap_meas)

def gen_cal_cap_res_test_data(Cal_test_matrix,noise_sd_t = 0.0001,noise_sd_v = 0.0001):
    t = np.linspace(0,1,300)
    tau = np.arange(0,365,3)
    C0 = 2.05
    R0 = 71.6e-3
    Volt_meas = []
    Temp_meas = []
    Res_meas = []
    Cap_meas = []
    Volt_noised = []
    Temp_noised = []
    SOC_meas = []
    Current_meas = []

    for V_sto, T_sto in Cal_test_matrix:
        TT_cal_noised = []
        VV_cal_noised = []
        SS_cal, TT_cal, VV_cal, II_cal, CC_cal, RR_cal = Cal_ageing_tests(t,tau,V_sto,T_sto,0.2,C0,R0) 
        for i in range(len(TT_cal)):
            TT_cal_noised.append(TT_cal[i] + np.random.normal(0,noise_sd_t,len(TT_cal[i])))
            VV_cal_noised.append(VV_cal[i] + np.random.normal(0,noise_sd_v,len(VV_cal[i])))
        Volt_meas.append(VV_cal)
        Temp_meas.append(TT_cal)
        Volt_noised.append(VV_cal_noised)
        Temp_noised.append(TT_cal_noised)
        Res_meas.append(RR_cal)
        Cap_meas.append(CC_cal)
        SOC_meas.append(SS_cal)
        Current_meas.append(II_cal)
    res = Kalman_Filter_Res(t,Temp_noised,Current_meas,0.0001)*1e3
    cap = Kalman_Filter_Cap(t,Volt_noised,Current_meas,res,0.0001)
    np.save('cal_cap_test.npy',cap/2.05)
    np.save('cal_res_test.npy',res/71.6e-3)
    np.save('cal_cap_test_true.npy',np.array(Cap_meas)/2.05)
    np.save('cal_res_test_true.npy',np.array(Res_meas)/71.6e-3)
    return res, cap, np.array(Res_meas),np.array(Cap_meas)
#%%


def gen_cyc_cap_res_training_data(cyc_train_matrix,noise_sd_t = 0.3,noise_sd_v = 0.01):
    t=np.linspace(0,24,3000)
    tau = np.arange(0,365,3)
    C0 = 2.05
    R0 = 71.6e-3
    Volt_meas = []
    Temp_meas = []
    Res_meas = []
    Cap_meas = []
    Volt_noised = []
    Temp_noised = []
    SOC_meas = []
    Current_meas = []
    Th_meas = []

    for DOD, V_av in cyc_train_matrix:
        Volt_storage = V_av
        T_en = -10
        Temp_storage = -10
        TT_cyc_noised = []
        VV_cyc_noised = []
        SS_cyc, TT_cyc, VV_cyc, II_cyc, CC_cyc, RR_cyc,Th = Cycling(t,Volt_storage, 
                                                                 Temp_storage, DOD, V_av, T_en,I,C0,R0) 
        for i in range(len(TT_cyc)):
            TT_cyc_noised.append(TT_cyc[i] + np.random.normal(0,noise_sd_t,len(TT_cyc[i])))
            VV_cyc_noised.append(VV_cyc[i] + np.random.normal(0,noise_sd_v,len(VV_cyc[i])))
        Volt_meas.append(VV_cyc)
        Temp_meas.append(TT_cyc)
        Volt_noised.append(VV_cyc_noised)
        Temp_noised.append(TT_cyc_noised)
        Res_meas.append(RR_cyc)
        Cap_meas.append(CC_cyc)
        SOC_meas.append(SS_cyc)
        Current_meas.append(II_cyc)
        Th_meas.append(Th)
    res = Kalman_Filter_Res(t,Temp_noised,Current_meas,4.1)*1e3
    cap = Kalman_Filter_Cap(t,Volt_noised,Current_meas,res,2.4)
    np.save('cyc_cap_train.npy',cap/2.05)
    np.save('cyc_res_train.npy',res/71.6e-3)
    np.save('cyc_cap_train_true.npy',np.array(Cap_meas)/2.05)
    np.save('cyc_res_train_true.npy',np.array(Res_meas)/71.6e-3)
    np.save('cyc_current_train.npy',np.array(Current_meas))
    np.save('cyc_th_train.npy',np.array(Th_meas))
    return res, cap,np.array(Res_meas),np.array(Cap_meas)

def gen_cyc_cap_res_test_data(cyc_test_matrix,noise_sd_t = 0.1,noise_sd_v = 0.01):
    t=np.linspace(0,24,3000)
    tau = np.arange(0,365,3)
    C0 = 2.05
    R0 = 71.6e-3
    Volt_meas = []
    Temp_meas = []
    Res_meas = []
    Cap_meas = []
    Volt_noised = []
    Temp_noised = []
    SOC_meas = []
    Current_meas = []
    Th_meas = []
    
    for DOD, V_av in cyc_test_matrix:
        Volt_storage = V_av
        T_en = -10
        Temp_storage = -10
        TT_cyc_noised = []
        VV_cyc_noised = []
        SS_cyc, TT_cyc, VV_cyc, II_cyc, CC_cyc, RR_cyc,Th = Cycling(t,Volt_storage, 
                                                                 Temp_storage, DOD, V_av, T_en,I,C0,R0) 
        for i in range(len(TT_cyc)):
            TT_cyc_noised.append(TT_cyc[i] + np.random.normal(0,noise_sd_t,len(TT_cyc[i])))
            VV_cyc_noised.append(VV_cyc[i] + np.random.normal(0,noise_sd_v,len(VV_cyc[i])))
        Volt_meas.append(VV_cyc)
        Temp_meas.append(TT_cyc)
        Volt_noised.append(VV_cyc_noised)
        Temp_noised.append(TT_cyc_noised)
        Res_meas.append(RR_cyc)
        Cap_meas.append(CC_cyc)
        SOC_meas.append(SS_cyc)
        Current_meas.append(II_cyc)
        Th_meas.append(Th)
    res = Kalman_Filter_Res(t,Temp_noised,Current_meas,1e3)*1e3
    cap = Kalman_Filter_Cap(t,Volt_noised,Current_meas,res,0.16)
    np.save('cyc_cap_test.npy',cap/2.05)
    np.save('cyc_res_test.npy',res/71.6e-3)
    np.save('cyc_cap_test_true.npy',np.array(Cap_meas)/2.05)
    np.save('cyc_res_test_true.npy',np.array(Res_meas)/71.6e-3)
    np.save('cyc_th_test.npy',np.array(Th_meas))
    np.save('cyc_current_test.npy',np.array(Current_meas))
    return res, cap,np.array(Res_meas), np.array(Cap_meas),Temp_noised,Volt_noised
#%%
cyc_test_matrix=cyc_aging_test_matrix_func()
cyc_train_matrix=cyc_aging_train_matrix_func()
cal_test_matrix=cal_aging_test_matrix_func()
cal_train_matrix=cal_aging_train_matrix_func()
#%%
np.save('cyc_test_matrix.npy',cyc_test_matrix)
np.save('cyc_train_matrix.npy',cyc_train_matrix)
np.save('cal_test_matrix.npy',cal_test_matrix)
np.save('cal_train_matrix.npy',cal_train_matrix)
#%%
tr = gen_cyc_cap_res_test_data(cyc_train_matrix,noise_sd_t = 0.2,noise_sd_v = 0.01)
#%%
te = gen_cal_cap_res_test_data(cal_train_matrix,noise_sd_t = 0.000001,noise_sd_v = 0.000001)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 15:03:33 2023

@author: compsysbio
"""
#%%
import numpy as np
from scipy.optimize import fsolve
from funcs_updated import *
#%%
t = np.linspace(0,24,3000)
tau = np.arange(0,365,3)
#%%

def Kalman_Filter_Res(t,T_stores,Current_Profile,noise_sd = 1e3):
    
    h=t[1]-t[0]
    def F2(T,I,R):
        h=t[1]-t[0]
        return (1/(c + h * 3600*U*SA)) * ( c * T + h*3600*(R * I**2 + U*SA * T_en))

    def H2(T,R,I,k):
        return T
    
    T_est_stores = []
    R_est_stores = []

     #ambient temperature

    c = 1760*0.045*1e-03

    #U = 0.1178*1e-03 #=kA/l
    
    U = 11*1e-03 #convective heat transfer coeficient (same as h = k/d, k= 0.2)
    
    d = 18e-3
    
    r = d/2
    
    l = 65e-3
    
    SA = 2*np.pi*r**2 + 2*np.pi*r*l
    
    for n in range(len(T_stores)):
    
        T_est_store = []
        R_est_store = []
        
        
    
        for j in range(len(tau)):
            T = T_stores[n][j]
            I = Current_Profile[n][j]
            
            T_en = T[0]
    
            var_e = noise_sd**2
            var_w = 0
            if (j==0):
                R_0 = 71.6e-03*1e-03
            else:
                R_0 = R_est_store[-1]
            #PR_0= (R_0-(71.6e-03*1e-03*(1+j/300)))**2
            PR_0 = 1e-5#1e-4
    
            T_0 = T[0]
            #PT_0=(T_0-20)**2
            PT_0 = 1#0.3#0.09#0.09#1#0.09
            
            
            PR_pred=[PR_0]
            R_pred=[R_0]
    
            PT_pred=[PT_0]
            T_pred=[T_0]
    
            PR_est=[PR_0]
            R_est=[R_0]
    
            PT_est=[PT_0]
            T_est=[T_0]
    
            A=np.ones(len(t))* c /(c + h * U*SA)
            B=np.zeros(len(t))
            KR=[0]
            CR=[0]
            KT=[0]
            CT=[0]
    
            DT=[1 for k in range(len(t))]
            DR=[1 for k in range(len(t))]
    
            Sto = [0]
    
            Pw=0
    
            n_meas = min(len(T),800)
            for k in range(1,n_meas):
    
                #Time update for the weight filter
                R_pred.append(R_est[k-1])
                PR_pred.append (PR_est[k-1])
    
                #Time update for the state filter
                T_pred.append(F2(T_est[k-1],I[k], R_pred[k]))
                PT_pred.append(A[k-1]*PT_est[k-1]*A[k-1] + B[k-1]*Pw*B[k-1])
    
                #Meas. update for the state filter
                CT.append(1)
    
                KT.append((PT_pred[k]*CT[k])*(CT[k]*PT_pred[k]*CT[k]+DT[k]*var_e*DT[k])**(-1))
                T_est.append(T_pred[k]+KT[k]*(T[k] - H2(T_pred[k],R_pred[k],I[k],k)))
                PT_est.append((1-KT[k]*CT[k])*PT_pred[k])
    
                #Measurement update for the weight filter
                Sto.append( I[k]**2 + c * (Sto[k-1]) *( 1 - KT[k-1]))                 
                CR.append((h*3600/(c + h *3600* U*SA)) * Sto[k])

                KR.append((PR_pred[k]*CR[k])*(CR[k]*PR_pred[k]*CR[k]+DR[k]*var_e*DR[k])**(-1))
                R_est.append(R_pred[k]+KR[k]*(T[k]-H2(T_pred[k],R_pred[k],I[k],k)))
                PR_est.append((1-KR[k]*CR[k])*PR_pred[k])
    
            T_est_store.append(T_est)
            R_est_store.append(R_est[-1])
        R_est_stores.append(R_est_store)
    return (np.array(R_est_stores))
#%%
#Kalman Filter for Capacity
def Kalman_Filter_Cap(t,V_stores,Current_Profile,R_est_stores,noise_sd=0.16):
    h=t[1]-t[0]
    def F1(S,I,Q):
        h=t[1]-t[0]
        return S + h * (I / Q)
    
    def H1(S,Q,I,R,k):
        return V_oc(S) + I * R 
    
    #S_est_stores = []
    Q_est_stores = []
    for n in range(len(V_stores)):
    
        S_est_store = []
        Q_est_store = []
        
    
        for j in range(len(tau)):
            
            #err=e[n][j]
            V = V_stores[n][j]
            I = Current_Profile[n][j]
            
            def V_oc2(S):
                return V_oc(S) - V[0]
            
            R = R_est_stores[n][j]
    
            var_e=noise_sd**2
            #var_w=0
    
            if (j==0):
                Q_0 = 2.05
            else:
                Q_0 = Q_est_store[-1]
            PQ_0= 0.1#1
    
            S_0 = fsolve(V_oc2, 0)[0] 
            PS_0= 0.1#1
            PQ_pred=[PQ_0]
            Q_pred=[Q_0]
    
            PS_pred=[PS_0]
            S_pred=[S_0]
    
            PQ_est=[PQ_0]
            Q_est=[Q_0]
    
            PS_est=[PS_0]
            S_est=[S_0]
    
            A=np.ones(len(t))
            B=np.zeros(len(t))
            KQ=[0]
            CQ=[0]
            KS=[0]
            CS=[0]
    
            DS=[1 for k in range(len(t))]
            DQ=[1 for k in range(len(t))]
    
            Sto = [0]
    
            Pw=0
    
            n_meas=min(len(I),800)
            for k in range(1,n_meas):
    
                #Time update for the weight filter
                Q_pred.append(Q_est[k-1])
                PQ_pred.append (PQ_est[k-1])
    
                #Time update for the state filter
                S_pred.append(F1(S_est[k-1],I[k-1], Q_pred[k]))
                PS_pred.append(A[k-1]*PS_est[k-1]*A[k-1] + B[k-1]*Pw*B[k-1])
    
                #Meas. update for the state filter
                CS.append(dV_oc_ds(S_pred[k]))
    
                KS.append((PS_pred[k]*CS[k])*(CS[k]*PS_pred[k]*CS[k]+DS[k]*var_e*DS[k])**(-1))
                S_est.append(S_pred[k]+KS[k]*(V[k] - H1(S_pred[k],Q_pred[k],I[k-1],R,k)))
                PS_est.append((1-KS[k]*CS[k])*PS_pred[k])
    
                #Meas. update for the weight filter
                Sto.append(dV_oc_ds(S_pred[k]) * (np.sum(np.array(I)[:k]) -
                                                  np.sum((np.array(KS)[1:k] * np.array(Sto)[1:k]))))                  
                CQ.append(-(h / Q_pred[k]**2) * Sto[k])
    
    
                KQ.append((PQ_pred[k]*CQ[k])*(CQ[k]*PQ_pred[k]*CQ[k]+DQ[k]*var_e*DQ[k])**(-1))
                Q_est.append(Q_pred[k]+KQ[k]*(V[k]-H1(S_pred[k],Q_pred[k],I[k-1],R,k)))
                PQ_est.append((1-KQ[k]*CQ[k])*PQ_pred[k])
    
            S_est_store.append(S_est)
            Q_est_store.append(Q_est[-1])
        Q_est_stores.append(Q_est_store)

    return np.array(Q_est_stores)

#%%

def ekf_Res_single(t,T,I,noise_sd = 1e3):
    
    h=t[1]-t[0]
    def F2(T,I,R):
        h=t[1]-t[0]
        return (1/(c + h * 3600*U*SA)) * ( c * T + h*3600*(R * I**2 + U*SA * T_en))

    def H2(T,R,I,k):
        return T
    
    T_est_stores = []
    R_est_stores = []

     #ambient temperature

    c = 1760*0.045*1e-03

    #U = 0.1178*1e-03 #=kA/l
    
    U = 11*1e-03 #convective heat transfer coeficient (same as h = k/d, k= 0.2)
    
    d = 18e-3
    
    r = d/2
    
    l = 65e-3
    
    SA = 2*np.pi*r**2 + 2*np.pi*r*l
    

    T_en = T[0]

    var_e = noise_sd**2
    var_w = 0

    R_0 = 71.6e-03*1e-03

    #PR_0= (R_0-(71.6e-03*1e-03*(1+j/300)))**2
    PR_0 = 1e-5#1e-4

    T_0 = T[0]
    #PT_0=(T_0-20)**2
    PT_0 = 1#0.3#0.09#0.09#1#0.09
    
    
    PR_pred=[PR_0]
    R_pred=[R_0]

    PT_pred=[PT_0]
    T_pred=[T_0]

    PR_est=[PR_0]
    R_est=[R_0]

    PT_est=[PT_0]
    T_est=[T_0]

    A=np.ones(len(t))* c /(c + h * U*SA)
    B=np.zeros(len(t))
    KR=[0]
    CR=[0]
    KT=[0]
    CT=[0]

    DT=[1 for k in range(len(t))]
    DR=[1 for k in range(len(t))]

    Sto = [0]

    Pw=0

    n_meas = min(len(T),800)
    for k in range(1,n_meas):

        #Time update for the weight filter
        R_pred.append(R_est[k-1])
        PR_pred.append (PR_est[k-1])

        #Time update for the state filter
        T_pred.append(F2(T_est[k-1],I[k], R_pred[k]))
        PT_pred.append(A[k-1]*PT_est[k-1]*A[k-1] + B[k-1]*Pw*B[k-1])

        #Meas. update for the state filter
        CT.append(1)

        KT.append((PT_pred[k]*CT[k])*(CT[k]*PT_pred[k]*CT[k]+DT[k]*var_e*DT[k])**(-1))
        T_est.append(T_pred[k]+KT[k]*(T[k] - H2(T_pred[k],R_pred[k],I[k],k)))
        PT_est.append((1-KT[k]*CT[k])*PT_pred[k])

        #Measurement update for the weight filter
        Sto.append( I[k]**2 + c * (Sto[k-1]) *( 1 - KT[k-1]))                 
        CR.append((h*3600/(c + h *3600* U*SA)) * Sto[k])

        KR.append((PR_pred[k]*CR[k])*(CR[k]*PR_pred[k]*CR[k]+DR[k]*var_e*DR[k])**(-1))
        R_est.append(R_pred[k]+KR[k]*(T[k]-H2(T_pred[k],R_pred[k],I[k],k)))
        PR_est.append((1-KR[k]*CR[k])*PR_pred[k])

    return (np.array(R_est)*1e3)
#%%
#Kalman Filter for Capacity
def ekf_Cap_single(t,V,I,R,noise_sd=0.16):
    h=t[1]-t[0]
    def F1(S,I,Q):
        h=t[1]-t[0]
        return S + h * (I / Q)
    
    def H1(S,Q,I,R,k):
        return V_oc(S) + I * R 
            
    def V_oc2(S):
        return V_oc(S) - V[0]
    

    var_e=noise_sd**2
    #var_w=0
    Q_0 = 2.05


    S_0 = fsolve(V_oc2, 0)[0] 
    PS_0= 0.1#1
    PQ_0= 0.1
    PQ_pred=[PQ_0]
    Q_pred=[Q_0]

    PS_pred=[PS_0]
    S_pred=[S_0]

    PQ_est=[PQ_0]
    Q_est=[Q_0]

    PS_est=[PS_0]
    S_est=[S_0]

    A=np.ones(len(t))
    B=np.zeros(len(t))
    KQ=[0]
    CQ=[0]
    KS=[0]
    CS=[0]

    DS=[1 for k in range(len(t))]
    DQ=[1 for k in range(len(t))]

    Sto = [0]

    Pw=0

    n_meas=min(len(I),800)
    for k in range(1,n_meas):

        #Time update for the weight filter
        Q_pred.append(Q_est[k-1])
        PQ_pred.append (PQ_est[k-1])

        #Time update for the state filter
        S_pred.append(F1(S_est[k-1],I[k-1], Q_pred[k]))
        PS_pred.append(A[k-1]*PS_est[k-1]*A[k-1] + B[k-1]*Pw*B[k-1])

        #Meas. update for the state filter
        CS.append(dV_oc_ds(S_pred[k]))

        KS.append((PS_pred[k]*CS[k])*(CS[k]*PS_pred[k]*CS[k]+DS[k]*var_e*DS[k])**(-1))
        S_est.append(S_pred[k]+KS[k]*(V[k] - H1(S_pred[k],Q_pred[k],I[k-1],R,k)))
        PS_est.append((1-KS[k]*CS[k])*PS_pred[k])

        #Meas. update for the weight filter
        Sto.append(dV_oc_ds(S_pred[k]) * (np.sum(np.array(I)[:k]) -
                                          np.sum((np.array(KS)[1:k] * np.array(Sto)[1:k]))))                  
        CQ.append(-(h / Q_pred[k]**2) * Sto[k])


        KQ.append((PQ_pred[k]*CQ[k])*(CQ[k]*PQ_pred[k]*CQ[k]+DQ[k]*var_e*DQ[k])**(-1))
        Q_est.append(Q_pred[k]+KQ[k]*(V[k]-H1(S_pred[k],Q_pred[k],I[k-1],R,k)))
        PQ_est.append((1-KQ[k]*CQ[k])*PQ_pred[k])



    return np.array(Q_est)
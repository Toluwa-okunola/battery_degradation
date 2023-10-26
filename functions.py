# -*- coding: utf-8 -*-
import numpy as np

from scipy.optimize import fsolve
from decimal import Decimal

#%%

#Parameters for Holistic Model for Resistance and Capacity:
C0 =2.05
R0 = 71.6e-03

def alpha_C(V,Temp):
    return (7.7543 * V - 23.75) * 10**6 * np.exp(-6976/(Temp+273))
def alpha_R(V,Temp): 
    return (5.27 * V - 16.32) * 10**5 * np.exp(-5986/(Temp+273))
def beta_C(V_av,DOD): 
    return 7.348 * 10**(-3) * (V_av - 3.667)**2 + 7.6 * 10 **(-4) + 4.801 * 10**(-3) * DOD
def beta_R(V_av,DOD): 
    return 2.153 * 10**(-4) * (V_av - 3.725)**2 + 1.521 * 10 **(-5) + 2.798 * 10**(-4) * DOD

def beta_C_V(V_av): 
    return 7.348 * 10**(-3) * (V_av - 3.667)**2
def beta_C_D(DOD): 
    return 7.6 * 10 **(-4) + 4.801 * 10**(-3) * DOD 

#%%

#Holistic Model for Resistance and Capacity:
   
#Charge Throughput    
def Throughput(t,tau,I_history):
    I_history = np.repeat(I_history,[3 for i in range(len(I_history))],0)
    ti=t
    if (tau == 0):
        return 0
    else:
        return np.array([np.trapz(abs(I_history[i]),t) for i in range(len(I_history))]).sum()
    
def Cap(t,tau, V, Temp, V_av, DOD,I_history,C0):
    C_nom = 2.05
    if (tau == 0):
        return C0
    elif (DOD==0):
        return C_nom*(C0/C_nom  - alpha_C(V, Temp)* tau**0.75)
    else:
        return C_nom*(C0/C_nom  - alpha_C(V, Temp)* tau**0.75 - beta_C(V_av, DOD)
                   * np.sqrt(Throughput(t,tau,I_history)))
    
def Cap_alt(t,tau, V, Temp, V_av, DOD,I_history,C0):
    C_nom = 2.05
    if (tau == 0):
        return C0
    elif (DOD==0):
        return C_nom*(C0/C_nom - alpha_C(V, Temp)* tau**0.75)
    else:
        return C_nom*(C0/C_nom  - alpha_C(V, Temp)* tau**0.75 - 
                   ((beta_C_V(V_av)+beta_C_D(DOD)) * np.sqrt(Throughput(t,tau,I_history)))/
                   (1 + beta_C_D(DOD) * np.sqrt(Throughput(t,tau,I_history))))

def Res(t,tau, V, Temp, V_av, DOD,I_history,R0):
    R_nom = 71.6e-3
    if (tau == 0):
        return R0
    elif (DOD==0):
        return R_nom*(R0/R_nom + alpha_R(V, Temp)* tau**0.75)
    else:
        return R_nom*(R0/R_nom + alpha_R(V, Temp)* tau**0.75 + beta_R(V_av,DOD)
                   * Throughput(t,tau,I_history))
    
def Res_cal(tau, V, Temp,R0):
    R_nom = 71.6e-3
    return R_nom*(R0/R_nom  + alpha_R(V, Temp)* tau**0.75)

def Cap_cal(tau, V, Temp,C0):
    C_nom = 2.05
    return C_nom*(C0/C_nom  - alpha_C(V, Temp)* tau**0.75)

tau = np.arange(0,365,3)
#%%

#OCV-SOV Curve

def V_oc(S):
    a = 3.3301439
    b = 2.28498918
    c = -7.3631483
    d = 12.22272519
    e = -8.1536875
    f = 1.84051118
    return (a + b*S + c*S**2 + d*S**3 + e*S**4 + f*S**5)

def dV_oc_ds(S):
    b = 2.28498918
    c = -7.3631483
    d = 12.22272519
    e = -8.1536875
    f = 1.84051118
    return b + 2 * c*S + 3* d*S**2 + 4 * e*S**3 + 5 * f*S**4

#%%
def I(t, S,V,V_max,V_min,cycle,R,C0):
    C0 =2.05
    c_rate = 1
    if cycle == 'charge':
        if(V < V_max):
            I_ = C0*c_rate
        else:
            I_ = (V_max-V_oc(S))/R
    if cycle == 'discharge':
        if (V>V_min):
            I_ = - C0*c_rate 
        else:
            I_ = (V_min-V_oc(S))/R
    if cycle == 'stop':
            I_ = 0
    return I_

def I_cal(t, S,V,V_max,V_min,cycle,R,C0):
    #C0 = 2.05
    c_rate = 0.2
    if cycle == 'charge':
        if(V < V_max):
            I_ = C0*c_rate
        else:
            I_ = (V_max-V_oc(S))/R
    if cycle == 'discharge':
        if (V>V_min):
            I_ = - C0*c_rate 
        else:
            I_ = (V_min-V_oc(S))/R
    if cycle == 'stop':
            I_ = 0
    return I_

#%%
tau=np.arange(0,365,3)
def Cycling(t,Volt_storage, Temp_storage, DOD, V_av, T_en,I,C0,R0):
    def V_oc2(S):
        return V_oc(S) - V_av
    
    R_nom=71.6e-3
    C_nom = 2.05
    S_max = fsolve(V_oc2, 0)[0] + DOD/2
    S_min = fsolve(V_oc2, 0)[0] - DOD/2
    
    V_max = 4.2#4.2
    V_min = 3.0#3.0
    
    
    S_stores =[]
    T_stores =[]
    V_stores = []
    I_stores = []
            
    
    S_store=[np.empty(len(t))]
    T_store=[np.empty(len(t))]
    C_store =[C0]
    R_store =[R0]
    V_store =[]
    I_store =[]   
    Th_store = [0]
    
    if (DOD == 0):
        
        for k in range(1,len(tau)+1):
            R = Res(t, tau[k-1], Volt_storage, Temp_storage, V_av, 
                    DOD,I_store[1:k],R0)
            C = Cap(t, tau[k-1], Volt_storage, Temp_storage, V_av, 
                    DOD,I_store[1:k],C0)
            C_store.append(C)    
            R_store.append(R)
        R_store = np.array(R_store)[1:]
        C_store = np.array(C_store)[1:]

                    
                
        return  (S_store, T_store,V_store,I_store,C_store,R_store)
    
    else:
                
        h = (t[1]-t[0])

        c = 1760*0.045

        #U = 0.1178 # = kA/l
        
        U = 11 #convective heat transfer coeficient (same as h = k/d, k= 0.2)
        
        d = 18e-3
        
        r = d/2
        
        l = 65e-3
        
        A = 2*np.pi*r**2 + 2*np.pi*r*l
        
        C = C_store[0]
        R = R_store[0]
        for k in range(1,len(tau)+1):
            if(k==1):
                T_av = T_en #just to avoid referencing before assignment
                C=C0
                
            Th = Throughput(t,tau[k-1],I_store[:k])
                
            C = Cap_alt(t, tau[k-1], V_av, T_av, V_av, 
                    DOD,I_store[:k],C0)

            if(k==1):
                R = R0
            else:
                R = R_nom*(R/R_nom - alpha_R(V_av, T_av)*(tau[k-2]**0.75 
                - tau[k-1]**0.75) - beta_R(V_av, DOD*C/C_nom)*(Th_store[k-1]- Th))
            #R0=R

            
            cycle = 'charge' 
            T = [T_en]
            S = [fsolve(V_oc2, 0)[0]]
            V_meas=[V_oc(S[0])]
            I_applied = [I(t[0],S[0],V_meas[0],V_max,V_min,cycle,R,C0)]
            
            
            for i in range(1,t.size): 

                S.append(S[-1] + 
                         h*(1/C)*I_applied[i-1])
                V_meas.append(V_oc(S[i]) + R*I_applied[i-1])
                T.append((1/(c + h * 3600* U*A)) * (c * T[-1] +
                            h * 3600 *(R * I(t[i],S[i],V_meas[i],V_max,V_min,cycle,R,C0)**2 
                                       + U * A * T_en)))
                
                
                I_applied.append(I(t, S[i],V_meas[i],V_max,V_min,cycle,R,C0))
                
                if abs(S[i]-S_max)<=1e-2:
                    cycle = 'discharge'
                    break
                
            half_cycle_point = i
                
            for i in range(half_cycle_point+1,t.size): 
                S.append(S[-1] + 
                         h*(1/C)*I_applied[i-1])
                
                V_meas.append(V_oc(S[i]) + R*I_applied[i-1])
                T.append((1/(c + h * 3600* U*A)) * (c * T[-1] +
                            h * 3600 *(R * I(t[i],S[i],V_meas[i],V_max,V_min,cycle,R,C0)**2 +
                                       U *A* T_en)))
                
                
                I_applied.append(I(t, S[i],V_meas[i],V_max,V_min,cycle,R,C0))
                
                if abs(S[i]-S_max)<=1e-2:
                    cycle = 'discharge'
                elif (abs(S[i]-S_min)<=1e-2):
                    cycle = 'charge'                  
                    
            S = np.array(S)
            T = np.array(T)
            I_applied = np.array(I_applied)

            V_meas = np.array(V_meas)
                    
            
                    
            S_store.append(S)    
            T_store.append(T)
            C_store.append(C)    
            R_store.append(R)
            V_store.append(V_meas)
            I_store.append(I_applied)
            Th_store.append(Th)
            
            T_av = np.mean(np.array(T_store)[1:])
  
                    
        S_store = np.array(S_store)[1:]
        T_store = np.array(T_store)[1:]
        V_store = np.array(V_store)
        I_store = np.array(I_store)
        R_store = np.array(R_store)[1:]
        C_store = np.array(C_store)[1:]
        Th_store = np.array(Th_store)[1:]
                
        return  (S_store, T_store,V_store,I_store,C_store,R_store,Th_store)
#%%
#Calendar Ageing Tests
tau=np.arange(0,365,3)
def Cal_ageing_tests(t,tau,Volt_storage,Temp_storage,DOD,C0,R0):       
    def V_oc2(S):
        return V_oc(S) - Volt_storage
    S_max = fsolve(V_oc2, 0)[0] + DOD/2
    S_min = fsolve(V_oc2, 0)[0] - DOD/2
    
    V_max = 4.2
    V_min = 3.0

    
    R_store = []
    C_store = []
    S_store=[]
    T_store=[]
    C_store =[]
    R_store =[]
    V_store =[]
    I_store =[]
    
       
    c = 1760*0.045
    
    #U = 0.1178
    
    U = 11 #convective heat transfer coeficient (same as h = k/d, k= 0.2)
    
    d = 18e-3
    
    r = d/2
    
    l = 65e-3
    
    A = 2*np.pi*r**2 + 2*np.pi*r*l
    
    if (Volt_storage<=3.7):
    
        for k in range(1,len(tau)+1):
            R = Res_cal(tau[k-1], Volt_storage, Temp_storage,R0)
            C = Cap_cal(tau[k-1], Volt_storage, Temp_storage,C0)
            C_store.append(C)    
            R_store.append(R)
    
        
        
            cycle = 'charge' 
            T_en = Temp_storage
            T = [T_en]
            S = [fsolve(V_oc2, 0)[0]]
            V_meas=[V_oc(S[0])]
            I_applied = [I_cal(t[0],S[0],V_meas[0],V_max,V_min,cycle,R,C0)]
        
            h = (t[1]-t[0])
            
            
            for i in range(1,t.size): 
        
                S.append(S[-1] + 
                         h*(1/C)*I_applied[i-1])
                V_meas.append(V_oc(S[i]) + R*I_applied[i-1])
                T.append((1/(c + h * 3600* U * A)) * (c * T[-1] +
                            h * 3600 *(R * I_cal(t[i],S[i],V_meas[i],V_max,V_min,
                                        cycle,R,C0)**2 + U * A * T_en)))
                
                
                I_applied.append(I_cal(t, S[i],V_meas[i],V_max,V_min,cycle,R,C0))
                
                if abs(S[i]-S_max)<=1e-3:
                    cycle = 'discharge'
                    break
                
            half_cycle_point = i
                
            for i in range(half_cycle_point+1,t.size): 
                S.append(S[-1] + 
                         h*(1/C)*I_applied[i-1])
                
                V_meas.append(V_oc(S[i]) + R*I_applied[i-1])
                T.append((1/(c + h * 3600 * U * A)) * (c * T[-1] +
                            h * 3600 *(R * I_cal(t[i],S[i],V_meas[i],V_max,V_min,
                                                 cycle,R,C0)**2 + U * A* T_en)))
                
                
                I_applied.append(I_cal(t, S[i],V_meas[i],V_max,V_min,cycle,R,C0))
                
                if (abs(S[i]-S[0])<=1e-3):
                    cycle = 'stop'
                    break                  
                    
            S = np.array(S)
            T = np.array(T)
            I_applied = np.array(I_applied)
        
            V_meas = np.array(V_meas)
            
            S_store.append(S[:250])    
            T_store.append(T[:250])
            #C_store.append(C)    
            #R_store.append(R)
            V_store.append(V_meas[:250])
            I_store.append(I_applied[:250])
    else:
        for k in range(1,len(tau)+1):
            R = Res_cal(tau[k-1], Volt_storage, Temp_storage,R0)
            C = Cap_cal(tau[k-1], Volt_storage, Temp_storage,C0)
            C_store.append(C)    
            R_store.append(R)

        
        
            cycle = 'discharge' 
            T_en = Temp_storage
            T = [T_en]
            S = [fsolve(V_oc2, 0)[0]]
            V_meas=[V_oc(S[0])]
            I_applied = [I_cal(t[0],S[0],V_meas[0],V_max,V_min,cycle,R,C0)]
        
            h = (t[1]-t[0])
    
            
            
            for i in range(1,t.size): 
        
                S.append(S[-1] + 
                         h*(1/C)*I_applied[i-1])
                V_meas.append(V_oc(S[i]) + R*I_applied[i-1])
                T.append((1/(c + h * 3600* U * A)) * (c * T[-1] +
                            h * 3600 *(R * I_cal(t[i],S[i],V_meas[i],V_max,V_min,
                                        cycle,R,C0)**2 + U * A * T_en)))
                
                
                I_applied.append(I_cal(t, S[i],V_meas[i],V_max,V_min,cycle,R,C0))
                
                if abs(S[i]-S_min)<=1e-3:
                    cycle = 'charge'
                    break
                
            half_cycle_point = i
                
            for i in range(half_cycle_point+1,t.size): 
                S.append(S[-1] + 
                         h*(1/C)*I_applied[i-1])
                
                V_meas.append(V_oc(S[i]) + R*I_applied[i-1])
                T.append((1/(c + h * 3600* U*A)) * (c * T[-1] +
                            h * 3600 *(R * I_cal(t[i],S[i],V_meas[i],V_max,V_min,
                                                 cycle,R,C0)**2 + U * A*T_en)))
                
                
                I_applied.append(I_cal(t, S[i],V_meas[i],V_max,V_min,cycle,R,C0))
                
                if (abs(S[i]-S[0])<=1e-3):
                    cycle = 'stop'
                    break                  
                    
            S = np.array(S)
            T = np.array(T)
            I_applied = np.array(I_applied)
        
            V_meas = np.array(V_meas)
            
            S_store.append(S[:250])    
            T_store.append(T[:250])

            V_store.append(V_meas[:250])
            I_store.append(I_applied[:250])
         
        S_store = np.array(S_store)
        T_store = np.array(T_store)
        V_store = np.array(V_store)
        I_store = np.array(I_store)
        R_store = np.array(R_store)
        C_store = np.array(C_store)
                     
    return  (S_store, T_store,V_store,I_store,C_store,R_store)
#%%
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]
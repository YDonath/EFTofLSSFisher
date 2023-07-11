import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.special import legendre
import seaborn as sns
from pathlib import Path
import sys
from classy import Class
import copy
import itertools
import pickle

kk = np.logspace(-5, 0, 200)
lnAsback=3.044
nsback=0.965
hback=0.673
Omback=0.317365
obback=0.02237
mnuback=0.1
Okback=0

def cosmicdicn(zpk,arr=np.array([0,0,0,0,0,0,0])):
    dlnAs=arr[0]*0.1
    dns=arr[1]*0.02
    dh=arr[2]*0.02
    dOm=arr[3]*0.04
    dob=arr[4]*0.02
    dmnu=arr[5]*0.25
    dOk=arr[6]*0.05
    return {'omega_b': obback*(1+dob),
          'omega_cdm':Omback*(1+dOm)*hback**2-obback*(1+dob)-mnuback*(1+dmnu)/93.14+2*Omback*hback**2*dh,
          'h': hback*(1+dh),
          'ln10^{10}A_s': lnAsback+dlnAs,
          'n_s': nsback*(1+dns),
          'output': 'mPk,mTk',
          'P_k_max_h/Mpc': 1,
          'z_pk': zpk,
          'N_ncdm':1,
          'N_ur': 2.0328,
          'm_ncdm':mnuback*(1+dmnu),#0.15#0.06
          'T_ncdm': 0.71611,
          'Omega_k':Okback+dOk
         }

def getD(dic):
    M = Class()
    M.set(dic)
    M.compute()
    zpk = dic['z_pk']
    
    # k in h/Mpc
    
    # P(k) in (Mpc/h)**3
    dd= M.scale_independent_growth_factor(dic['z_pk'])
    return  dd


def getpk(dic):
    M = Class()
    M.set(dic)
    M.compute()
    # k in h/Mpc
    zpk = dic['z_pk']
    
    # P(k) in (Mpc/h)**3
    Pk = np.array([M.pk(ki*M.h(), zpk)*M.h()**3 for ki in kk])
    return  Pk

def getf(dic):
    M = Class()
    M.set(dic)
    M.compute()
    zpk = dic['z_pk']
    
    # k in h/Mpc
    
    # P(k) in (Mpc/h)**3
    ff= M.scale_independent_growth_factor_f(dic['z_pk'])
    return  ff

def getT(dic):
    M = Class()
    M.set(dic)
    M.compute()
    zpk = dic['z_pk']

    Dg = M.scale_independent_growth_factor(zpk)/((1+100)*M.scale_independent_growth_factor(100))
    Possionpre =  (3.e5)**2/(1.5*100.**2*M.Omega0_m())

    
    tdic = M.get_transfer(zpk)
    kpsi= tdic['k (h/Mpc)']
    Tpsi = tdic['psi']/tdic['psi'][0]
    return kpsi,Tpsi*kpsi**2*Possionpre*Dg

def getsymplectic(n):
    arrs1 = np.identity(n)
    arrsfull = np.zeros((2*n+1,n))
    for i in range(n):
        arrsfull[2*i+1]=arrs1[i]
        arrsfull[2*i+2]=-arrs1[i]
    return arrsfull.astype(int)


def getalldics(zpk):
    allp=[]
    fullsymp=getsymplectic(7)
    for i in range(len(fullsymp)):
        allp.append(cosmicdicn(zpk,fullsymp[i]))
    return np.array(allp)

def getallpks(dics):
	allp=[]
	allp.append(kk)
	for i in range(len(dics)):
		allp.append(getpk(dics[i]))
	return np.array(allp)

def getallTs(dics):
    allT=[]
    for i in range(len(dics)):
        ks,Ta = getT(dics[i])
        allT.append(ks)
        allT.append(Ta)
    return np.array(allT)

def getallfs(dics):
	allf=[]
	for i in range(len(dics)):
		allf.append(getf(dics[i]))
	return np.array(allf)

names = np.array(['fid','lnAs_pl','lnAs_min','ns_pl','ns_min','h_pl','h_min','Om_pl','Om_min','ob_pl','ob_min','nu_pl','nu_min','Ok_pl','Ok_min'])

def savepk(pks,folder):
	os.mkdir(folder)
	for i in range(len(pks)-1):
	    np.savetxt(os.path.join(folder, 'pk_'+names[i] + '.dat'), np.transpose(np.array([pks[0],pks[i+1]])))

def savepkandf(pks,fs,folder):
	os.mkdir(folder)
	for i in range(len(pks)-1):
	    np.savetxt(os.path.join(folder, 'pk_'+names[i] + '.dat'), np.transpose(np.array([pks[0],pks[i+1]])))
	np.savetxt(os.path.join(folder, 'allfs.dat'),fs)

def savepkandfandT(pks,Ts,fs,folder):
    os.mkdir(folder)
    for i in range(len(pks)-1):
        np.savetxt(os.path.join(folder, 'pk_'+names[i] + '.dat'), np.transpose(np.array([pks[0],pks[i+1]])))
        np.savetxt(os.path.join(folder, 'Tk_'+names[i] + '.dat'), np.transpose(np.array([Ts[2*i],Ts[2*i+1]])))
    np.savetxt(os.path.join(folder, 'allfs.dat'),fs)

def allsaved(dics,folder):
	fs = getallfs(dics)
	pks = getallpks(dics)
	savepkandf(pks,fs,folder)

def allsavedT(dics,folder):
    fs = getallfs(dics)
    pks = getallpks(dics)
    Ts = getallTs(dics)
    savepkandfandT(pks,Ts,fs,folder)


def getDandf(ztab):
    growthD =[]
    growthR =[]
    for i in range(len(ztab)):
        growthD.append(getD(cosmicdicn(ztab[i])))
        growthR.append(getf(cosmicdicn(ztab[i])))
    print(growthD)
    print(growthR)



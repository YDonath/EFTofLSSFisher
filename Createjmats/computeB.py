

import numpy as np
import os
import sys
import h5py
from functools import lru_cache
import pandas as pd
from Jfunc_cython_v6 import computeJ as J
# from computeJ_Guido import computeJ as J
import gmpy2 as gm
from gmpy2 import *
import time
from tqdm import trange
import time
from scipy.io import mmread, mmwrite
gm.get_context().precision = 200
gm.get_context().allow_complex = True


###################################################################################################
#Preliminaries

def saver(path,dat):
    with h5py.File(path,'w') as h5_file:
        h5_file.create_dataset("jmat", data=dat)
    h5_file.close()

def outpufolder(k1,k2,k3):
    return str("%.5f" % float(k1)) + '_' + str("%.5f" % float(k2)) + '_' + str("%.5f" % float(k3)) +'_.h5'
    
numl = 16
#############################################################################################################
#B222 75s

ctab222 = np.loadtxt('diagramtabs/B222ctab.dat', dtype = object).astype(float).astype(int)
n222=len(ctab222)

def jmatB222(k1,k2,k3):

    k11 = mpfr(k1)
    k22 = mpfr(k2)
    k33 = mpfr(k3)
    
    k12 = k11**2
    k22 = k22**2
    k32 = k33**2

    
    ltriantable = np.zeros((n222,numl,numl,numl),dtype=np.float64)
    for i1 in reversed(range(numl)):
        for i2 in reversed(range(numl)):
            for i3 in reversed(range(numl)):
                for i in range(n222):
                    ltriantable[i,i1,i2,i3] = J(ctab222[i,0], ctab222[i,1], ctab222[i,2], i1, i2, i3, k12, k22, k32)
    return ltriantable
    
###################################################################################################################
#B3211 15s
 
ctab3211 = np.loadtxt('diagramtabs/B3211ctab.dat', dtype = object).astype(float).astype(int)
n3211=len(ctab3211) 
    
def jmatB3211(k1,k2,k3):

    k11 = mpfr(k1)
    k22 = mpfr(k2)
    k33 = mpfr(k3)
    
    k12 = k11**2
    k22 = k22**2
    k32 = k33**2

    ltriantable = np.zeros((n3211,numl,numl,6),dtype=float)
    for i1 in reversed(range(numl)):
        for i2 in reversed(range(numl)):
            for i in range(n3211):
                ltriantable[i,i1,i2,0] = J(ctab3211[i,0], ctab3211[i,1], ctab3211[i,2], i1, -1, i2, k12,k22, k32)
                ltriantable[i,i1,i2,1] = J(ctab3211[i,0], ctab3211[i,1], ctab3211[i,2], i1, -1, i2, k12,k32, k22)	
                ltriantable[i,i1,i2,2] = J(ctab3211[i,0], ctab3211[i,1], ctab3211[i,2], i1, -1, i2, k22,k12, k32)
                ltriantable[i,i1,i2,3] = J(ctab3211[i,0], ctab3211[i,1], ctab3211[i,2], i1, -1, i2, k22,k32, k12)	
                ltriantable[i,i1,i2,4] = J(ctab3211[i,0], ctab3211[i,1], ctab3211[i,2], i1, -1, i2, k32,k12, k22)
                ltriantable[i,i1,i2,5] = J(ctab3211[i,0], ctab3211[i,1], ctab3211[i,2], i1, -1, i2, k32,k22, k12)
    return ltriantable    
    
##################################################################################################################
#B3212 0.1s
 
ctab3212 = np.loadtxt('diagramtabs/B3212ctab.dat', dtype = object).astype(float).astype(int)
n3212=len(ctab3212) 

def jmatB3212(k1,k2,k3):
    k11 = mpfr(k1)
    k22 = mpfr(k2)
    k33 = mpfr(k3)
    
    k12 = k11**2
    k22 = k22**2
    k32 = k33**2


    ltriantable = np.zeros((n3212,numl,6),dtype=float)
    for i1 in reversed(range(numl)):
        for i in range(n3212):
            ltriantable[i,i1,0] = J(ctab3212[i,0], ctab3212[i,1], ctab3212[i,2], i1, -1, -1, k12, k22, k32)
            ltriantable[i,i1,1] = J(ctab3212[i,0], ctab3212[i,1], ctab3212[i,2], i1, -1, -1, k12,k32, k22)	
            ltriantable[i,i1,2] = J(ctab3212[i,0], ctab3212[i,1], ctab3212[i,2], i1, -1, -1, k22,k12, k32)
            ltriantable[i,i1,3] = J(ctab3212[i,0], ctab3212[i,1], ctab3212[i,2], i1, -1, -1, k22,k32, k12)	
            ltriantable[i,i1,4] = J(ctab3212[i,0], ctab3212[i,1], ctab3212[i,2], i1, -1, -1, k32,k12, k22)
            ltriantable[i,i1,5] = J(ctab3212[i,0], ctab3212[i,1], ctab3212[i,2], i1, -1, -1, k32,k22, k12)
    return ltriantable
                
                
############################################################################################################
#B411 1.5ss
 
ctab411 = np.loadtxt('diagramtabs/B411ctab.dat', dtype = object).astype(float).astype(int)
n411=len(ctab411) 

def jmatB411(k1,k2,k3):
    k11 = mpfr(k1)
    k22 = mpfr(k2)
    k33 = mpfr(k3)
    
    k12 = k11**2
    k22 = k22**2
    k32 = k33**2


    ltriantable = np.zeros((n411,numl,3),dtype=float)
    for i1 in reversed(range(numl)):
        for i in range(n411):
            if ctab411[i,1] != 0:
                ltriantable[i, i1,0] = J(ctab411[i,0], ctab411[i,2], ctab411[i,4], i1, -1, -1, k12, k22, k32)
                ltriantable[i, i1,1] = J(ctab411[i,0], ctab411[i,2], ctab411[i,4], i1, -1, -1, k22, k32, k12)
                ltriantable[i, i1,2] = J(ctab411[i,0], ctab411[i,2], ctab411[i,4], i1, -1, -1, k32, k12, k22)
                
            elif ctab411[i,3] != 0:
                ltriantable[i, i1,0] = J(ctab411[i,0], ctab411[i,2], ctab411[i,4], -1, i1, -1, k12, k22, k32)
                ltriantable[i, i1,1] = J(ctab411[i,0], ctab411[i,2], ctab411[i,4], -1, i1, -1, k22, k32, k12)
                ltriantable[i, i1,2] = J(ctab411[i,0], ctab411[i,2], ctab411[i,4], -1, i1, -1, k32, k12, k22)

            elif ctab411[i,5] != 0:
                ltriantable[i, i1,0] = J(ctab411[i,0], ctab411[i,2], ctab411[i,4], -1, -1, i1, k12, k22, k32)
                ltriantable[i, i1,1] = J(ctab411[i,0], ctab411[i,2], ctab411[i,4], -1, -1, i1, k22, k32, k12)
                ltriantable[i, i1,2] = J(ctab411[i,0], ctab411[i,2], ctab411[i,4], -1, -1, i1, k32, k12, k22)
    return ltriantable
                
###############################################################################################################
#Final powerspectrum function approx 40s per jmat
def jmatsB(k1,k2,k3):
    path222 = 'jmats/B222/'+outpufolder(k1,k2,k3)
    if not(os.path.isfile(path222)):
        saver(path222,jmatB222(k1,k2,k3))
    path3211 = 'jmats/B3211/'+outpufolder(k1,k2,k3)
    if not(os.path.isfile(path3211)):
        saver(path3211,jmatB3211(k1,k2,k3))
    path3212 ='jmats/B3212/'+outpufolder(k1,k2,k3) 
    if not(os.path.isfile(path3212)):
        saver(path3212,jmatB3212(k1,k2,k3))
    path411 = 'jmats/B411/'+outpufolder(k1,k2,k3)
    if not(os.path.isfile(path411)):
        saver(path411,jmatB411(k1,k2,k3))

def jmatsB2(triang):
    return jmatsB(triang[0],triang[1],triang[2])

def jmatsB_from_arr(arr):
    for i in range(len(arr)):
        jmatsB2(arr[i])
        print(i)

def jmatsB_from_file(pathfile):
    arr = mmread(pathfile)
    jmatsB_from_arr(arr)









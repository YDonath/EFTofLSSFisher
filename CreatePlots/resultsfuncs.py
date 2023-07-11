import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from scipy.io import mmread
from tabulate import tabulate
from getdist import plots, gaussian_mixtures
from scipy.special import erf, erfinv



paramindex = ["lnAs", "ns", "h", "Om", "ob", "logmn","Ok",
"fNLloc", "fNLeq", "fNLorth", "logb1", "c2", "b3", "b4", "c4", "b6",
"b7", "b8", "b9", "b10", "b11", "Bc1", "Bc2", "Bc3", "Bc4", "Be1",
"Be2", "ce2", "Bc5", "Bc6", "Bc7", "Bc8", "Bc9", "Bc10", "Bc11",
"Bc12", "Bc13", "Bc14", "Bd1", "Bd2", "Bd3", "Be3", "Be4", "Be5",
"Be6", "Be7", "Be8", "Be9", "Be10","Be11", "Be12"]


fnlloc = ["fNLloc"]
fnleq = ["fNLeq"]
fnlorth = ["fNLorth"]
biasindex = ["logb1", "c2", "b3", "b4", "c4", "b6",
"b7", "b8", "b9", "b10", "b11", "Bc1", "Bc2", "Bc3", "Bc4", "Be1",
"Be2", "ce2", "Bc5", "Bc6", "Bc7", "Bc8", "Bc9", "Bc10", "Bc11",
"Bc12", "Bc13", "Bc14", "Bd1", "Bd2", "Bd3", "Be3", "Be4", "Be5",
"Be6", "Be7", "Be8", "Be9", "Be10","Be11", "Be12"]
biasindexstfix = ["logb1", "c2", "b3", "b4", "c4", "b6",
"b7", "b8", "b9", "b10", "b11", "Bc1", "Bc2", "Bc3", "Bc4",
"Be2", "ce2", "Bc5", "Bc6", "Bc7", "Bc8", "Bc9", "Bc10", "Bc11",
"Bc12", "Bc13", "Bc14", "Bd2", "Bd3", "Be3", "Be4", "Be5",
"Be6", "Be7", "Be8", "Be9", "Be10","Be11", "Be12"]

def multiindex(arr1,arr2):
    res = []
    for i in range(len(arr2)):
        new = list(np.where(np.array(arr1) == arr2[i])[0])   
        for j in range(len(new)):
            res.append(new[j])
    return list(np.array(res).flatten())

def parampos(arr): return multiindex(paramindex,arr)


paramindex4sky = ["lnAs", "ns", "h", "Om", "ob", "logmn","Ok",
"fNLloc", "fNLeq", "fNLorth"]+list(np.transpose(np.array([np.array(biasindex),np.array(biasindex),np.array(biasindex),np.array(biasindex)])).flatten())

def parampos4sky(arr): return multiindex(paramindex4sky,arr)


Paramlabels = ["ln(10^{10}A_s)", "n_s", "h", "\Omega_m", "\omega_b", "\log(\sum_im_{\\nu_i}/eV)","\Omega_k","f_{NL}^{loc}", "f_{NL}^{eq}", "f_{NL}^{orth}", "\log(b_1)", "c_2", "b_3", "b_4", "c_4", "b_6", "b_7", "b_8", "b_9",
"b_{10}", "b_{11}", "Bc_1", "Bc_2", "Bc_3", "Bc_4",
"Be_1", "Be_2", "ce_2", "Bc_5", "Bc_6", "Bc_7", "Bc_8", "Bc_9", "Bc_{10}",
"Bc_{11}", "Bc_{12}", "Bc_{13}", "Bc_{14}", "Bd_1", "Bd_2", "Bd_3", "Be_3", "Be4",
"Be_5", "Be_6", "Be_7", "Be_8", "Be_9", "Be_{10}", "Be_{11}", "Be_{12}"]

background = [3.044, 0.965, 0.673, 0.317365, 0.02237, np.log(0.1),0,0,0,0,np.log(1.9391), 
1.1447, -0.374, 0.1276, -0.2895, -0.3487, 0.2234, -0.2972, 0.0148, 
0.0428, 0.0356, 5.4605, -1.5439, 1.3081, -0.4785, 1.6922, 0.9091, 
0.5548, 0.1054, 0.8713, -0.4555, 0.4442, -0.4152, -0.65, -0.0881, 
-0.3731, -0.1642, -0.1958, 5.4, -0.721, -0.4266, 0.0737, -0.1381, 
6.0673, -0.0932, -0.9702, 0.2617, 0.2638, -0.1526, 0.427, -0.4322, 
9.6]

def geterror(cov):
    errors = []
    for i in range(len(cov)):
        errors.append(np.sqrt(cov[i,i]))
    return errors

def around2(arr,n):
    rounded = np.zeros(arr.shape)
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            rounded[i,j]=np.format_float_positional(arr[i,j], precision=n, unique=False, fractional=False, trim='k')
    return rounded


def gettable(errors,cosmopar,labs):
    errtab = errors
    for i in range(len(labs)):
        errtab[i].insert(0,labs[i])
    errtab = np.array(errtab).T.tolist()
    errtab[0].insert(0,r"\sigma")
    for i in range(len(cosmopar)):
        errtab[i+1].insert(0,cosmopar[i])
    return np.array(errtab).T.tolist()

def combinesamps(samplist):
    samps = []
    for j in range(len(samplist)):
        for i in range(len(samplist[j])):
            samps.append(samplist[j][i])
    return samps

def mnbounds(sig,mu,n=1):
    if n==1:
        lowerq=0.16
        upperq=0.84
    if n==2:
        lowerq=0.025
        upperq=0.975
    if n==3:
        lowerq=0.005
        upperq=0.995
    mub=mu
    return [np.exp(np.sqrt(2)*sig*erfinv(2*lowerq-1))*mub-mub,np.exp(np.sqrt(2)*sig*erfinv(2*upperq-1))*mub-mub]   


def getPlot(base,cosmovar,plotvar,labs,save=0,legs=12,axs=12,las=13,mnon=0):
    samps = []
    errors = []
    mnerrors=[]
    cosmopar = list(np.array(Paramlabels)[parampos(plotvar)])
    for i in range(len(base)):
        if(len(base[i])==174):
            freeparams = parampos4sky(cosmovar)
            newparams =list(np.array(paramindex4sky)[freeparams])
            Plotparams = multiindex(newparams,plotvar)    
        elif(len(base[i])==51):
            freeparams = parampos(cosmovar)
            newparams =list(np.array(paramindex)[freeparams])
            Plotparams = multiindex(newparams,plotvar)
        else:
            print('wrong dim')
            
        Fisher = base[i][freeparams,:][:,freeparams]
        
        fishcov = np.linalg.inv(Fisher)[Plotparams,:][:,Plotparams]
        sampl = np.random.multivariate_normal(list(np.array(background)[parampos(plotvar)]), fishcov, size=int(1e5))
        samps.append(MCSamples(samples=sampl, names=cosmopar, labels = cosmopar))
        errors.append(geterror(fishcov))
        if (mnon):
            muback= background[5]
            mupos = list(np.where(np.array(newparams)[np.array(Plotparams)] == 'logmn')[0])[0]  
            mnerrors.append(mnbounds(geterror(fishcov)[mupos],np.exp(muback)))
    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = axs
    g.settings.lab_fontsize = las
    g.settings.legend_fontsize = legs
    g.triangle_plot(samps,cosmopar, filled=True, legend_labels = labs)
    if (mnon):
        print(mnerrors)
    if(save):
        plt.savefig(save+'.pdf')
        with open(save+'.txt', 'w') as f:
            f.write(tabulate(gettable(around2(np.array(errors),2).tolist(),cosmopar,labs),tablefmt="latex")) 

def getError(base,cosmovar,plotvar,labs,save=0):
    samps = []
    errors = []
    cosmopar = list(np.array(Paramlabels)[parampos(plotvar)])
    for i in range(len(base)):
        if(len(base[i])==174):
            freeparams = parampos4sky(cosmovar)
            newparams =list(np.array(paramindex4sky)[freeparams])
            Plotparams = multiindex(newparams,plotvar)    
        elif(len(base[i])==51):
            freeparams = parampos(cosmovar)
            newparams =list(np.array(paramindex)[freeparams])
            Plotparams = multiindex(newparams,plotvar)
        else:
            print('wrong dim')
            
        Fisher = base[i][freeparams,:][:,freeparams]
        fishcov = np.linalg.inv(Fisher)[Plotparams,:][:,Plotparams]
        sampl = np.random.multivariate_normal(list(np.array(background)[parampos(plotvar)]), fishcov, size=int(1e5))
        samps.append(MCSamples(samples=sampl, names=cosmopar, labels = cosmopar))
        errors.append(geterror(fishcov))
    return errors

def printerr(a):
    return print(around2(np.array(a),4))


def getSamps(base,cosmovar,plotvar,labs):
    samps = []
    cosmopar = list(np.array(Paramlabels)[parampos(plotvar)])
    for i in range(len(base)):
        if(len(base[i])==174):
            freeparams = parampos4sky(cosmovar)
            newparams =list(np.array(paramindex4sky)[freeparams])
            Plotparams = multiindex(newparams,plotvar)    
        elif(len(base[i])==51):
            freeparams = parampos(cosmovar)
            newparams =list(np.array(paramindex)[freeparams])
            Plotparams = multiindex(newparams,plotvar)
        else:
            print('wrong dim')
            
        Fisher = base[i][freeparams,:][:,freeparams]
        
        fishcov = np.linalg.inv(Fisher)[Plotparams,:][:,Plotparams]
        sampl = np.random.multivariate_normal(list(np.array(background)[parampos(plotvar)]), fishcov, size=int(1e5))
        samps.append(MCSamples(samples=sampl, names=cosmopar, labels = cosmopar))
    return samps ,cosmopar,labs


def getPlotfromSamps(samps,cosmopar,labs,save=0,legs=12,axs=12,las=13):
    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = axs
    g.settings.lab_fontsize = las
    g.settings.legend_fontsize = legs
    g.triangle_plot(samps,cosmopar, filled=True, legend_labels = labs)
    if(save):
        plt.savefig(save+'.pdf')

def getError(base,cosmovar,plotvar,labs,save=0):
    samps = []
    errors = []
    cosmopar = list(np.array(Paramlabels)[parampos(plotvar)])
    for i in range(len(base)):
        if(len(base[i])==174):
            freeparams = parampos4sky(cosmovar)
            newparams =list(np.array(paramindex4sky)[freeparams])
            Plotparams = multiindex(newparams,plotvar)    
        elif(len(base[i])==51):
            freeparams = parampos(cosmovar)
            newparams =list(np.array(paramindex)[freeparams])
            Plotparams = multiindex(newparams,plotvar)
        else:
            print('wrong dim')
            
        Fisher = base[i][freeparams,:][:,freeparams]
        fishcov = np.linalg.inv(Fisher)[Plotparams,:][:,Plotparams]
        sampl = np.random.multivariate_normal(list(np.array(background)[parampos(plotvar)]), fishcov, size=int(1e5))
        samps.append(MCSamples(samples=sampl, names=cosmopar, labels = cosmopar))
        errors.append(geterror(fishcov))
    return errors

def fnlerrors(base,cosmovar,plotvar,save=0):
    errors = []
    for i in range(len(base)):
        if(len(base[i])==174):
            freeparams = parampos4sky(cosmovar)
            newparams =list(np.array(paramindex4sky)[freeparams])
            Plotparams = multiindex(newparams,plotvar)    
        elif(len(base[i])==51):
            freeparams = parampos(cosmovar)
            newparams =list(np.array(paramindex)[freeparams])
            Plotparams = multiindex(newparams,plotvar)
        else:
            print('wrong dim')
        Fisher = base[i][freeparams,:][:,freeparams]
        fishcov = np.linalg.inv(Fisher)[Plotparams,:][:,Plotparams]
        errors.append(np.sqrt(fishcov[0,0]))
        
    if (save):
        np.savetxt(save+'.txt',np.array(errors))
    print(errors)



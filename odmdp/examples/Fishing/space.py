#Bijection between difference space and bounded space
import numpy as np

#Magnitude W function
def Mw(X):
    return (np.sum(abs(X)) + abs(np.sum(X)))/2.0

#Magnitude B function
def Mb(X):
    return np.max(X)

#Convert difference space to bounded space
def DB(X):
    sh = X.shape
    cX = X.reshape([1,X.size])
    wX = DW(cX)
    bX = WB(wX)
    return bX

#Convert bounded space to difference space
def BD(X,sh):
    wX = BW(X)
    dX = WD(wX)
    return dX.reshape(sh)
    
#Convert from difference space to weight space
def DW(X):
    wX = -np.delete(X,0)
    return wX

#Convert from weight space to bounded space
def WB(X):
    return Mw(X)/Mb(X)*X

#Convert from bounded space to weight space
def BW(X):
    return Mb(X)/Mw(X)*X

#Convert from weight space to difference space
def WD(X):
    dX = np.append([np.sum(X)],-X)
    return dX
    

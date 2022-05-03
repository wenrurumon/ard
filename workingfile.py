
import sys
import os
import numpy as np
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from numpy import diag as diag
from numpy import ones as ones
from numpy import trace as trace
from numpy import identity as identity
from numpy import log as log
from numpy import outer as outer
from numpy import cumprod as cumprod
from numpy.linalg import det as det, svd
from numpy.linalg import inv as inv
from numpy.linalg import slogdet as slogdet
from numpy.linalg import cholesky as chol
from numpy.random import normal as normal
from numpy.random import multivariate_normal as multivariate_normal
from scipy.stats import matrix_normal

float_formatter = "{:.3e}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

def heatmap(V,title='',scale=1,vmax=0,vmin=-1,cmap='RdBu_r'): 
    if vmax==0:
        vmax = np.max(abs(V))*scale
    sns.heatmap(pd.DataFrame(V),cmap=cmap,vmax=vmax,vmin=vmin*vmax)
    plt.title(title)
    plt.show()
def ispd(A):
    M = np.matrix(A)
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False
def aspd(A):
    A0 = A
    i = 0
    while not ispd(A):
        i = i+1
        A = A + np.spacing(np.linalg.norm(A)) * identity(A.shape[0])
    return (A,A-A0,i)
def gendata(D,M,N,K):
    X = normal(0.0, 1.0, size=(M,N))
    W2 = normal(0.0, 1.0, size=(D,M))
    K = list(range(0,M,K))
    W = np.zeros(W2.shape)
    W[:,K] = W2[:,K]
    V = np.identity(D)
    e = multivariate_normal([0.0] * D, V, N).T
    Y = W @ X + 1 * e
    b0 = (Y @ X.T) @ np.linalg.inv(X @ X.T)
    V = (Y - b0 @ X) @ (Y - b0 @ X).T / N
    return (X,Y,V,W)
def llh(X,Y,V,K):
    D,N = Y.shape
    S = inv(identity(N) - X.T @ inv(X @ X.T + inv(K)) @ X)
    return -N/2 * cumprod(slogdet(V))[-1] - D/2 * cumprod(slogdet(S))[-1] - 0.5 * trace(inv(V) @ Y @ inv(S) @ Y.T)
def train(X,Y,V0,K0,maxit):
    D,N = Y.shape
    M = X.shape[0]
    K = np.identity(M)
    invXXK = inv(X @ X.T + inv(K))
    V = 1/N * (Y @ Y.T - Y @ X.T @ invXXK @ X @ Y.T)
    for i in range(maxit):
        print(i,'@l =',llh(X,Y,V,K))
        invXXK = inv(X @ X.T + inv(K))
        V = 1/N * (Y @ Y.T - Y @ X.T @ invXXK @ X @ Y.T)
        invV = inv(V)
        M_W = Y @ X.T @ invXXK
        V_W = V
        K_W = invXXK
        K = diag((np.sum(np.multiply(V,invV))*diag(K_W) + diag(M_W.T @ invV @ M_W))/D)
    return (V,K)
def inv2(x):
    invx = inv(x)
    invx[range(x.shape[0]),range(x.shape[0])] = 0
    return invx
def fit(X,Y,maxit,epsV,epsK,cutX,maxit2,epsK2):
    #Initial
    D,N = Y.shape
    M = X.shape[0]
    X0 = X
    Y0 = Y
    M0 = M
    #X Reduction
    if(cutX*N<M):
        K = np.identity(M)
        K0 = K
        XX = aspd(X @ X.T)[0]
        YY = aspd(Y @ Y.T)[0]
        YX = Y @ X.T
        XY = YX.T
        for i in range(maxit2):
            print(i,'Interection for X Dimension Reduction')
            invXXK = inv(XX + inv(K))
            YXSXY = aspd(YX @ invXXK @ XY)[0]
            V = 1/N * (YY - YXSXY)
            invV = inv(V)
            K = diag((np.sum(np.multiply(V,invV))*diag(invXXK) + diag((YX @ invXXK).T @ invV @ YX @ invXXK))/D)
            K0 = K
            if np.max(abs(K-K0))<epsK2:
                break
        sel = diag(K) > np.quantile(diag(K),1-min(1,cutX*N/M))
        X = X0[np.array(range(M))[sel],:]
        M = X.shape[0]
    else:
        sel = [True] * M
    #Initial
    K = np.identity(M)
    XX = aspd(X @ X.T)[0]
    YY = aspd(Y @ Y.T)[0]
    YX = Y @ X.T
    XY = YX.T
    invXXK = inv(XX + inv(K))
    YXSXY = aspd(YX @ invXXK @ XY)[0]
    V = 1/N * (YY - YXSXY)
    V0 = V
    K0 = K
    loss0 = -np.inf
    #Interection
    for i in range(maxit):
        loss = llh(X,Y,V,K)
        if(loss<loss0):
            print("Warning loss<loss0")
            converge = -1
            # break
        invXXK = inv(XX + inv(K))
        YXSXY = aspd(YX @ invXXK @ XY)[0]
        V = 1/N * (YY - YXSXY)
        invV = inv(V)
        K = diag((np.sum(np.multiply(V,invV))*diag(invXXK) + diag((YX @ invXXK).T @ invV @ YX @ invXXK))/D)
        if i>0 and np.max(abs(V-V0))<epsV and np.max(abs(K-K0))<epsK and loss>loss0:
            print(i,'@l =',loss,' dV =',np.max(abs(V-V0)),' dK =',np.max(abs(K-K0)))
            converge = 1
            break
        else:
            print(i,'@l =',loss,' dV =',np.max(abs(V-V0)),' dK =',np.max(abs(K-K0)))
            converge = 0
            V0 = V
            K0 = K
            loss0 = loss
    #Result
    alpha = np.array([0.0]*M0)
    alpha[sel] = diag(K)
    return (V,alpha,converge)

##################################################
# 4 Scales
##################################################

#Load datafile

os.chdir('/Users/wenrurumon/Documents/postdoc/ard')
filename_genetic = ["RNA.csv"]
filename_micro = ["Metabolome.csv","Bioelectrcity.csv","Proteome.csv","Cell.csv","Physiology & Biochemistry .csv"]
filename_structure=["DXA.csv","Image.csv","Skin and accessory detection.csv","Ultrasound.csv"]
filename_function=["Sense organ.csv","Sleep monitoring.csv","Health questionnaire.csv","Constitutional measurement.csv","Psychology.csv","TCM.csv","Voice.csv"]

data_genetic = []
for k in filename_genetic:
    data_genetic.append(pd.read_csv(k))
data_micro = []
for k in filename_micro:
    data_micro.append(pd.read_csv(k))
data_structure = []
for k in filename_structure:
    data_structure.append(pd.read_csv(k))
data_function = []
for k in filename_function:
    data_function.append(pd.read_csv(k))

data_genetic = pd.concat(data_genetic,axis=1)
data_micro = pd.concat(data_micro,axis=1)
data_structure = pd.concat(data_structure,axis=1)
data_function = pd.concat(data_function,axis=1)

#Model to go

def model(X,Y,maxit,epsV,epsK,cutX,maxit2,epsK2,file):
    (V,K,Converge) = fit(np.mat(X).T,np.mat(Y).T,maxit,epsV,epsK,cutX,maxit2,epsK2)
    V = pd.DataFrame(inv2(V))
    V.columns = V.index = Y.columns
    K = pd.DataFrame(K,index=X.columns)
    return (V,K,Converge,file)

rlt = []
rlt.append(model(data_micro,data_structure,3000,1e-6,1e-4,0.8,100,1e-2,'m2s'))
rlt.append(model(data_structure,data_function,3000,1e-6,1e-4,0.8,100,1e-2,'s2f'))
rlt.append(model(data_function,data_structure,3000,1e-6,1e-4,0.8,100,1e-2,'f2s'))
rlt.append(model(data_structure,data_micro,3000,1e-6,1e-4,0.8,100,1e-2,'s2m'))

writer = pd.ExcelWriter('result0503.xlsx')
for i in rlt:
    i[0].to_excel(writer,sheet_name=i[3]+'_V.xlsx')
    i[1].to_excel(writer,sheet_name=i[3]+'_K.xlsx')
writer.save()


##################################################
# Debug
##################################################

X_ = np.mat(data_structure).T
Y_ = np.mat(data_micro).T

#Debug
X = X_
Y = Y_
maxit = 10000
epsV = epsK = 1e-5
cutX = 0.8
maxit2 = 100
epsK2 = 1e-2

#Initial
D,N = Y.shape
M = X.shape[0]
X0 = X
Y0 = Y
M0 = M

#X Reduction

K = np.identity(M)
K0 = K
XX = aspd(X @ X.T)[0]
YY = aspd(Y @ Y.T)[0]
YX = Y @ X.T
XY = YX.T
invXXK = inv(XX + inv(K))
Syx2 = YX @ invXXK @ XY

G = chol(aspd(invXXK)[0])
L = chol(aspd(Syx2)[0])
L2 = G2 = np.zeros((max(G.shape[0],L.shape[0]),max(G.shape[0],L.shape[0])))
G2[0:G.shape[0],0:G.shape[0]] = G
L2[0:L.shape[0],0:L.shape[0]] = L

L2 @ inv(G2)

V = 1/N * (YY - YX @ invXXK @ XY)

#X Reduction
if(cutX*N<M):
    K = np.identity(M)
    K0 = K
    XX = aspd(X @ X.T)[0]
    YY = aspd(Y @ Y.T)[0]
    YX = Y @ X.T
    XY = YX.T
    for i in range(maxit2):
        print(i,'Interection for X Dimension Reduction')
        invXXK = inv(XX + inv(K))
        V = 1/N * (YY - YX @ invXXK @ XY)
        invV = inv(V)
        print(ispd(invV))
        K = diag((np.sum(np.multiply(V,invV))*diag(invXXK) + diag((YX @ invXXK).T @ invV @ YX @ invXXK))/D)
        K0 = K
        if np.max(abs(K-K0))<epsK2:
            break
    sel = diag(K) > np.quantile(diag(K),1-min(1,cutX*N/M))
    X = X0[np.array(range(M))[sel],:]
    M = X.shape[0]
else:
    sel = [True] * M
    
#Initial
K = np.identity(M)
XX = aspd(X @ X.T)[0]
YY = aspd(Y @ Y.T)[0]
YX = Y @ X.T
XY = YX.T
invXXK = inv(XX + inv(K))
V = 1/N * (YY - YX @ invXXK @ XY)
V0 = V
K0 = K
loss0 = -np.inf
#Interection
for i in range(maxit):
    loss = llh(X,Y,V,K)
    if(loss<loss0):
        print("Warning loss<loss0")
        converge = -1
        break
    invXXK = inv(XX + inv(K))
    V = 1/N * (YY - YX @ invXXK @ XY)
    invV = inv(V)
    K = diag((np.sum(np.multiply(V,invV))*diag(invXXK) + diag((YX @ invXXK).T @ invV @ YX @ invXXK))/D)
    if i>0 and np.max(abs(V-V0))<epsV and np.max(abs(K-K0))<epsK and loss>loss0:
        print(i,'@l =',loss,' dV =',np.max(abs(V-V0)),' dK =',np.max(abs(K-K0)))
        converge = 1
        break
    else:
        print(i,'@l =',loss,' dV =',np.max(abs(V-V0)),' dK =',np.max(abs(K-K0)))
        converge = 0
        V0 = V
        K0 = K
        loss0 = loss
#Result
alpha = np.array([0.0]*M0)
alpha[sel] = diag(K)
return (V,alpha,converge)
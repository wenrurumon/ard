import sys
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

def fit(X,Y,maxit,epsV,epsK,cutX,maxit2):
    #Debug
    # X = X_
    # Y = Y_
    # maxit = 10000
    # epsV = 1e-6
    # epsK = 1e-6
    # cutX = 0.8
    # maxit2 = 100
    #Initial
    D,N = Y.shape
    M = X.shape[0]
    X0 = X
    Y0 = Y
    M0 = M
    #X Reduction
    if(cutX*N<M):
        K = np.identity(M)
        XX = aspd(X @ X.T)[0]
        YY = Y @ Y.T
        YX = Y @ X.T
        XY = YX.T
        for i in range(maxit2):
            print(i,'Interection for X Dimension Reduction')
            invXXK = inv(XX + inv(K))
            V = 1/N * (YY - YX @ invXXK @ XY)
            invV = inv(V)
            K = diag((np.sum(np.multiply(V,invV))*diag(invXXK) + diag((YX @ invXXK).T @ invV @ YX @ invXXK))/D)
        sel = diag(K) > np.quantile(diag(K),1-min(1,cutX*N/M))
        X = X0[np.array(range(M))[sel],:]
        M = X.shape[0]
    else:
        sel = [True] * M
    #Initial
    K = np.identity(M)
    XX = aspd(X @ X.T)[0]
    YY = Y @ Y.T
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
        invXXK = inv(XX + inv(K))
        V = 1/N * (YY - YX @ invXXK @ XY)
        invV = inv(V)
        K = diag((np.sum(np.multiply(V,invV))*diag(invXXK) + diag((YX @ invXXK).T @ invV @ YX @ invXXK))/D)
        if i>0 and np.max(V-V0)<epsV and np.max(K-K0)<epsK and loss>loss0:
            print(i,'@l =',loss,' dV =',np.max(abs(V-V0)),' dK =',np.max(abs(K-K0)))
            break
        else:
            print(i,'@l =',loss,' dV =',np.max(abs(V-V0)),' dK =',np.max(abs(K-K0)))
            V0 = V
            K0 = K
            loss0 = loss
    #Result
    alpha = np.array([0.0]*M0)
    alpha[sel] = diag(K)
    K = diag(alpha)
    return (V,K)

(X_,Y_,V_,W_) = gendata(D=20,M=150,N=100,K=10)
(Vh,Kh) = fit(X_,Y_,10000,1e-6,1e-6,0.8,10)
(Vx,Kx) = train(X_,Y_,0,0,10000)

heatmap((inv2(V_)),'Actual V')
heatmap((inv2(Vh)),'Predict V Hu')
heatmap((inv2(Vx)),'Predict V Xiaojun')
heatmap(W_.T,'Actual W')
heatmap(diag(Kh),'Predict K Hu')
heatmap(diag(Kx),'Predict K Xiaojun')

#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import decomposition 

mean = [0,0]   
cov = [[10.5,13], [13, 30]]
x_o = [18, 20, 21, 24]
y_o = [-22, -20, -20, -15]
x = np.random.multivariate_normal(mean, cov, 30) #N=30 datapoints

X = np.array(x)
x_out = [[18, -22], [20, -20], [21, -20], [24, -15]] #creating outliers to concatenate in a figure
X_o = np.array(x_out)
X = np.concatenate([X, X_o]).T
print(type(X))
print(X.shape)
L = 50


def l1pca_SBF_rank1_simplified(X,L):
    X = np.array(X)
    N = X.shape[1]
    max_iter = 50
    iteration = max_iter
    delta = np.zeros((1,N))  #initializing row vector
    obj_val = 0              # initializing the objective fucntion's value
    for l in range(L):       # no of initializations of b vector
        b = (np.random.rand(N,1)>0.5).astype('double')*2-1  # randomly initializing vector b (+1/-1) values
        for iteration in range(max_iter):
            for i in range(N):
                bi = b
                bi = np.delete(b, i)
                Xi = X
                Xi = np.delete(X, i, axis=1)
                scal_mult = np.multiply(-4,b[i])                
                delta[:,i] = float(np.multiply(scal_mult, np.matmul(np.matmul(X[:,i:i+1].T, Xi), bi)))
            val = -np.sort(-delta)  #sort the delta and find the bit that leads to big value
            ID = np.argsort(-delta)
            if val[:,0]>0:    #if highest increase is positive 
                b[ID[0]]=-b[ID[0]]  #extracting only those vectors which have positive value and flip the corresponding bit 
            else:
                break
            tmp = np.linalg.norm(np.matmul(X, b)) #calculate objective function's value
        if tmp > obj_val:              #if larger than old objective function, keep updating the function until it reqches the best value
            obj_val = tmp
            bopt = b
            l_best = l
    x_bopt = np.matmul(X, bopt)
    Qprop = x_bopt/np.linalg.norm(x_bopt)
    Bprop = bopt
    print(Qprop.shape, Bprop.shape)
    return Qprop,  Bprop, iteration, l_best
qprop, bprop, iteration, l_best = l1pca_SBF_rank1_simplified(X,L) #calling the function

# to calculate the slope of the L1-PCA generated values
a = qprop[1]/qprop[0]

#to calculate the inbuilt L2-PCA using the inbuilt function
pca = decomposition.PCA(n_components=1)
pca.fit(X.T)
X_L2 = pca.transform(X.T).T   #taking transpose of the matrix
qprop_L2 = pca.components_
slope = qprop_L2[0,1]/qprop_L2[0,0]   #calculating slope of the array for L2-PCA

linspace = list(range(-250,250,1))   #creating the range of -25 to 25 with 0.1 change in linear space
linspace = [x/10 for x in linspace]
y_L1 = a*linspace         
y_L2 = np.array([slope])*linspace

#creating a figure to create a subplot
fig = plt.figure()
ax = fig.add_subplot(111) #adding subplot in the figure to fit all the datapoints and outliers
ax.plot(x, 'ro', color = 'blue') # datapoints in blue
ax.axis('equal')
ax.plot(x, 'ro', color = 'blue') # datapoints in blue
ax.axis('equal')
ax.scatter(x_o,y_o,label='outlier', color='red', s=25, alpha = 0.9) #outliers in red
ax.plot(linspace, y_L1, color='green', label='L1-PCA') #to plot L1PCA component
ax.plot(linspace, y_L2, color='darkviolet', label='L2-PCA')
plt.show()







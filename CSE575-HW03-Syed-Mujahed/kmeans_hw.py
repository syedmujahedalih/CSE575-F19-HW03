# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:17:22 2019
CSE 575,Fall'19 HW3
Question-2: Implementation of KMeans Algorithm
@author: syedm
"""

import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt


class Kmeans:
    def __init__(self,X,K):
        self.X=X
        self.shp = X.shape
        self.Output={}
        self.Centroids=np.array([]).reshape(self.X.shape[1],0)
        self.K=K
        self.m=self.X.shape[0]
        self.l,self.n=self.shp
        self.loss = 0
        
    def kprob(self,X,K):
        i=rd.randint(0,X.shape[0])
        Centroid_temp=np.array([X[i]])
        print("Centroids=",Centroid_temp)
        for k in range(1,K):
            D=np.array([]) 
            for x in X:
                D=np.append(D,np.min(np.sum((x-Centroid_temp)**2)))
            prob=D/np.sum(D)
            cummulative_prob=np.cumsum(prob)
            r=rd.random()
            i=0
            for j,p in enumerate(cummulative_prob):
                if r<p:
                    i=j
                    break
            Centroid_temp=np.append(Centroid_temp,[X[i]],axis=0)
        return Centroid_temp.T
    
    def fit(self,n_iter):
        
        self.Centers=self.kprob(self.X,self.K)
        
        for n in range(n_iter):
            EuclidianDistance=np.array([]).reshape(self.m,0)
            for k in range(self.K):
                tempDist=np.sum((self.X-self.Centroids[:,k])**2,axis=1)
                EuclidianDistance=np.c_[EuclidianDistance,tempDist]
            C=np.argmin(EuclidianDistance,axis=1)+1
            self.loss = 0
            for i in range(len(EuclidianDistance)):
              self.loss += EuclidianDistance[i][C[i]-1]
            Y={}
            for k in range(self.K):
                Y[k+1]=np.array([]).reshape(self.n,0)
            for i in range(self.m):
                Y[C[i]]=np.c_[Y[C[i]],self.X[i]]
        
            for k in range(self.K):
                Y[k+1]=Y[k+1].T
            for k in range(self.K):
                self.Centers[:,k]=np.mean(Y[k+1],axis=0)
                
            self.Output=Y
            
    
    def predict(self):
        return self.Output,self.loss
      
data = pd.read_csv("CSE575-HW03-Data.csv")
data = np.array(data)

km = Kmeans(data,2)
ft = km.fit(50)
op,loss = km.predict()
class_1 = op[1]
class_1 = class_1[:,0:2]
class_2 = op[2]
class_2 = class_2[:,0:2]
plt.scatter(class_1[:,0],class_1[:,1],color='red')
plt.scatter(class_2[:,0],class_2[:,1],color='green')
plt.title('KMeans Clustering for k=2')
plt.show()


deviate = []
K = [2,3,4,5,6,7,8,9]
for k in K:
  km = Kmeans(data,k)
  ft = km.fit(50)
  op,loss = km.predict()
  deviate.append(loss)
plt.plot(K, deviate, '*-',color='turquoise')
plt.xlabel('Values of k')
plt.ylabel('Objective function')
plt.title('Objective Function as a function of "k"')
plt.show()

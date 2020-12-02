# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:31:29 2019
CSE 575,Fall'19 HW3
Question-4 part 2: Implementing an alternative to K-Means
@author: syedm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class newKmeans:
    
    def __init__(self,X,K):
        self.X=X
        self.shp = X.shape
        self.Output={}
        self.l,self.n=self.shp
        self.K=K
        for k in range(self.K):
            self.Centers=self.X[np.random.randint(self.X.shape[0], size=self.n), :]
            self.Centers = self.Centers.T
        self.m=self.X.shape[0]
        
        
    def fit(self,n_iter):
        
        for n in range(n_iter):
            EucDist=np.array([]).reshape(self.m,0)
            for k in range(self.K):
                tempDist=np.sum((np.abs(self.X-self.Centers[:,k])),axis=1)
                EucDist=np.c_[EucDist,tempDist]
            C=np.argmin(EucDist,axis=1)+1
            Yhat={}
            for k in range(self.K):
                Yhat[k+1]=np.array([]).reshape(self.n,0)
            for i in range(self.m):
                Yhat[C[i]]=np.c_[Yhat[C[i]],self.X[i]]
        
            for k in range(self.K):
                Yhat[k+1]=Yhat[k+1].T
            for k in range(self.K):
            	
            	ops = self.Centers
            	self.Centers[:,k]=self.calCenter(Yhat[k+1],k+1)
            	if(np.array_equal(ops,self.Centers)):
            		break 
                
                
            self.Output=Yhat
            
    
    def predict(self):
        return self.Output

    def calCenter(self,b,k):
    	n = b.shape[0]
    	op = {}
    	for i in range(0,n):
    		temp = b[i]
    		op[i] = np.sum((np.abs(temp-b[i,:])))
    	res = np.argmin(op)
    	return b[res]


    
data = pd.read_csv("CSE575-HW03-Data.csv")
data = np.array(data)
km_new = newKmeans(data,2)
ft = km_new.fit(500)
op = km_new.predict()

class_1 = op[1]
class_1 = class_1[:,0:2]
class_2 = op[2]
class_2 = class_2[:,0:2]
#class_3 = op[3]
#class_3 = class_3[:,0:2]
plt.scatter(class_1[:,0],class_1[:,1],color='red',label='class 1')
plt.scatter(class_2[:,0],class_2[:,1],color='green',label='class 2')
#plt.scatter(class_3[:,0],class_3[:,1],color='blue',label='class 3')
plt.legend()
plt.title('Alternative KMeans, k=2')
plt.show()
print("Center of Cluster 1 is ",km_new.Centers[:,0])
print("Center of Cluster 2 is ",km_new.Centers[:,1])
#print("Center of Cluster 3 is ",km_new.Centroids[:,2])
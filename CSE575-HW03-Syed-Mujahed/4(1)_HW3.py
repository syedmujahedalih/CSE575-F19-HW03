# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:31:29 2019
CSE 575,Fall'19 HW3
Question-4 part 1: Fixed point iteration method
@author: syedm
"""


import numpy as np
import pandas as pd

class fpIter:
    
    def __init__(self,X):
        self.x=X
        self.Output={}
        self.center=np.zeros((1,self.x.shape[1]))
        
        
    
    def Iter(self,n_iter):
        for i in range(n_iter):
            print("iteration number "+str(i))
            print("Center ",self.center)
            dist = np.abs(self.center-self.x)
            quo = np.zeros(dist.shape)
            np.divide(self.x, dist,out=quo, where=dist != 0)
            numerator = np.sum(quo,axis=0)
            j = np.ones(dist.shape)
            quo1 =  np.zeros(dist.shape)
            np.divide(j, dist,out=quo1, where=dist != 0)
            denominator = np.sum(quo1,axis=0)
            xnew = np.zeros((1,13))
            np.divide(numerator, denominator,out=xnew, where=denominator != 0)
            if(self.center in self.x and i!=0):
                break
            else:
                
                self.center = xnew
    
            
    
    def predict(self):
        return self.Output
    
data = pd.read_csv("CSE575-HW03-Data.csv")
data = np.array(data)
fp = fpIter(data)
fp.Iter(150)
center=fp.center
print(center)



# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:16:12 2019
CSE 575,Fall'19 HW3
Question-2: Implementation of kNN Algorithm
@author: syedm
"""

import mnist
from matplotlib import pyplot as plt
import numpy as np
import random
import numpy.matlib
from collections import Counter
random.seed(42)
np.random.seed(42)


train_images = mnist.train_images()
train_images = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
train_labels = mnist.train_labels()
train_labels = train_labels.reshape(-1,1)

test_images = mnist.test_images()
test_images = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))
test_labels = mnist.test_labels()
test_labels = test_labels.reshape(-1,1)

X_train = np.asarray(train_images).astype(np.float32)
y_train = np.asarray(train_labels).astype(np.int32)
X_test = np.asarray(test_images).astype(np.float32)
y_test = np.asarray(test_labels).astype(np.int32)

class kNN():

    def _init_(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    
    
    def Eucdistance(self, X):
        dot_product = np.dot(X, self.X_train.T)
        sum_square_tst = np.square(X).sum(axis = 1)
        sum_square_trn = np.square(self.X_train).sum(axis = 1)
        dists = np.sqrt(-2 * dot_product + sum_square_trn + np.matrix(sum_square_tst).T)
        print(dists.shape)
        return(dists)

    def predict(self, X, k=1):
        dists = self.distance(X)

        
        num_test = dists.shape[0]
        y_predicted = np.zeros(num_test)

        for i in range(num_test):
            k_closest_labels = []
            labels = self.y_train[np.argsort(dists[i,:])].flatten()
            # find k nearest lables
            k_closest_labels = labels[:k]
            c = Counter(k_closest_labels)
            y_predicted[i] = c.most_common(1)[0][0]

        return(y_predicted)


k_vals = [1, 3, 5, 10, 20, 30, 40, 50, 60]
dict1 = {}
acc = []
for k in k_vals:
  batch_size = 3000
  classifier = kNN()
  classifier.train(X_train, y_train)
  predictions = []
  for i in range(int(len(X_test)/(batch_size))):
      print("Computing batch " + str(i+1) + "/" + str(int(len(X_test)/batch_size)) + "...")
      pred = classifier.predict(X_test[i * batch_size:(i+1) * batch_size], k)
      predictions = predictions + list(pred)
  print("Completed predicting the test data.")
  print(len(predictions))
  y_test.shape
  cnt=0
  for i in range(len(predictions)):
    if y_test[i]==predictions[i]:
      cnt+=1
  acc.append((cnt/len(predictions)))
  print(cnt/len(predictions))
  dict1[k] = float(cnt/len(predictions)) 
  

plt.plot(k_vals, acc,'*-',color='turquoise')
plt.xlabel('Value of k')
plt.ylabel('Test Accuracy')
plt.title('Accuracy vs Value of k')
plt.show()
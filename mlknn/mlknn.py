# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:02:00 2019

@author: Konstantinos Pliakos


Implementation of the MLkNN algorithm [1].


[1] Zhang, M. L., & Zhou, Z. H. (2007). ML-KNN: A lazy learning approach to 
multi-label learning. Pattern recognition, 40(7), 2038-2048.


"""
import numpy as np
import numpy.matlib
from sklearn.neighbors import NearestNeighbors
#from ..exceptions import NotFittedError


class MLKNN:
       
    def __init__(self, smooth=1.0, threshold=0.5, n_jobs = 1):
        self.threshold = threshold # the threshold applied on the ouput probabilites
       # self.num = num
        self.smooth = smooth # the smoothing parameter, see [1] 
        self.n_jobs = n_jobs # common sklearn parameter for multi-threading 


    def fit(self, X, Y, num=5, distance_matrix = None, neighbors=None):
        """Training of the MLKNN. The function finds the nearest neighbors using the kNN implementation of scikit-learn. 
        It then computes the prior and conditional probabilities.
        
        If the user needs/wants to use another k-NN method, he/she can provide another NN matrix (i.e., neighbors) with shape [n_samples, n_NNs].  

        The user can provide a distance matrix instead (i.e., distance_matrix).
        
        
        Parameters
        ----------
        X : matrix of shape = [n_samples, n_features]
        (i.e., the feature matrix)
                
        Y : matrix of shape = [n_samples, n_outputs]
        (i.e., the label/output matrix)
        
        num: the number of nearest neighbors (default=5)
        
        neighbors: A nearest neighbor (NN) matrix with shape [n_samples, n_NNs]
        
        distance_matrix: set different than None if X is a distance matrix
        
        
        Returns
        -------
        self : object
            Returns self.
        """        

        if (neighbors is None) and (distance_matrix is None):
            self.neigh = NearestNeighbors(num,n_jobs=1)
            self.neigh.fit(X)
            self.neighbors = self.neigh.kneighbors(X, num, return_distance=False)
        elif distance_matrix is not None: 
            self.neighbors = np.argsort(X,1)
        else:
            self.neighbors = neighbors
#        self.neighbors = np.delete(neighbors,0,axis=1)
            
            
#        self.smooth = 1.0
        nb_train,nb_label = Y.shape

        #Computing prior and priorn
        temp_c = (Y==1).sum(0)
        self.prior = (self.smooth+temp_c) / (self.smooth*2 + nb_train)
        self.priorn = 1.0-self.prior
    
        temp_c = np.zeros([num+1,nb_label])
        temp_nc = np.zeros([num+1,nb_label])
        
        for i in range(nb_train):
            neighbor_labels = np.zeros([num,Y.shape[1]])
            for j in range(num):
                neighbor_labels[j,:] = Y[self.neighbors[i][j],:]
            temp = (neighbor_labels == 1).sum(0)
            for j in range(nb_label):
                if Y[i,j]==1:
                    temp_c[temp[j],j] = temp_c[temp[j],j] + 1
                else:
                    temp_nc[temp[j],j] = temp_nc[temp[j],j] + 1
        
        temp1 = temp_c.sum(0)
        temp2 = temp_nc.sum(0)
        self.cond=(self.smooth+temp_c) /(self.smooth*(num+1) + np.matlib.repmat(temp1,num+1,1))
        self.condn=(self.smooth+temp_nc) /(self.smooth*(num+1) + np.matlib.repmat(temp2,num+1,1))
        
        return self
        
        
        
    def predict(self,Xt,Y,num,distance_matrix_t = None,neighbors=None):
        """Training of the MLKNN. The function finds the nearest neighbors using the kNN implementation of scikit-learn. 
        It then computes the prior and conditional probabilities.
        
        In case the user needs/wants to use another k-NN method, he/she can provide another NN matrix with shape [n_samples, n_NNs].  
        
        Parameters
        ----------
        Xt : matrix of shape = [n_samples (test), n_features]
        (i.e., the feature matrix)
                
        Y : matrix of shape = [n_samples (train), n_outputs]
        (i.e., the label/output matrix)
        
        num: the number of nearest neighbors (default=5)
        
        neighbors: A nearest neighbor (NN) matrix with shape [n_samples (test), n_NNs].
        
        distance_matrix_t:  set different than None if X is a distance matrix
        
        Returns
        -------
        preds : The predicted labels (after threshold application)
        prob_outputs : The output probabilites
        """   


        try:
            self.condn
        except:
            raise NameError("Estimator not fitted.")

        if (neighbors is None) and (distance_matrix_t is None):
            self.neighbors_test = self.neigh.kneighbors(Xt, num, return_distance=False) 
        elif distance_matrix_t is not None: 
            self.neighbors_test = np.argsort(Xt,1)
        else:
            self.neighbors_test = neighbors
            
        nb_train,nb_labels = Y.shape
        nb_test = Xt.shape[0]
        
        
        prob_outputs = np.zeros([nb_test,nb_labels]) 
        for i in range(nb_test):
            neighbor_labels = np.zeros([num,Y.shape[1]])
            for j in range(num):
                neighbor_labels[j,:] = Y[self.neighbors_test[i][j],:]
            temp = (neighbor_labels == 1).sum(0)
            for j in range(nb_labels):
                p1 = self.prior[j]*self.cond[temp[j],j]
                p2 = self.priorn[j]*self.condn[temp[j],j]
                if (p1 + p2==0):
                    prob_outputs[i,j] = self.prior[j]
                else:
                    prob_outputs[i,j] = p1 / (p1 + p2)      
        preds = np.zeros([nb_test,nb_labels])
        preds[prob_outputs >= self.threshold] = 1
        preds[prob_outputs < self.threshold] = 0
        
        return preds,prob_outputs
    
    
    
    
    
    
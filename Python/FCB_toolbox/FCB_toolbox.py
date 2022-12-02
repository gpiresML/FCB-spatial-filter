# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 21:35:42 2022
version Python 3.8.8

FCB_spatial_filters
FCB_projections

@author: Gabriel Pires, June 2022
"""


import numpy as np

"""
Obtains FCB statistical spatial filters maximizing the discrimnation between two classes 
Inputs:
z1: trials class 1   (format channels x time samples x trials 
z2: trials class 2
th: regularization parameter
Outputs:
U1: eigenvectors (spatial filters odered by relevance)
V1: eigenvalues

Implementation of "Fisher Criterion Beamformer (FCB)" according to paper below (section 3.2.2):

Gabriel Pires, Urbano Nunes and  Miguel Castelo-Branco (2011), "Statistical Spatial Filtering for 
a P300-based BCI: Tests in able-bodied, and Patients with Cerebral Palsy and Amyotrophic Lateral 
Sclerosis", Journal of Neuroscience Methods, Elsevier, 2011, 195(2), 
Feb. 2011: doi:10.1016/j.jneumeth.2010.11.016
https://www.sciencedirect.com/science/article/pii/S0165027010006503?via%3Dihub

algorithm developed by Gabriel Pires 02/2011

Python Code implemented in June 2022
"""
def FCB_spatial_filters(z1,z2,th):
    Mean1 = np.mean(z1,axis=2)
    Mean2 = np.mean(z2,axis=2)

    Cov1=np.zeros((np.size(z1,0),np.size(z1,0),np.size(z1,2)))
    Cov2=np.zeros((np.size(z2,0),np.size(z2,0),np.size(z2,2)))
    
    for i in np.arange(np.size(z1,2)):     #for each trial in class 1 
        aux1=(z1[:,:,i]-Mean1) @ np.transpose(z1[:,:,i]-Mean1)
        Cov1[:,:,i]=aux1 / (np.trace(aux1));            #normalized spatial covariance per trial 

    for i in np.arange(np.size(z2,2)):     #for each trial in class 1 
        aux2=(z2[:,:,i]-Mean2) @ np.transpose(z2[:,:,i]-Mean2)
        Cov2[:,:,i]=aux2 / (np.trace(aux2));
        
    p1=np.size(z1,2)/(np.size(z1,2)+np.size(z2,2))
    p2=np.size(z2,2)/(np.size(z1,2)+np.size(z2,2))
    
    Covavg1=np.sum(Cov1,2); #covariances sum class 1  
    Covavg2=np.sum(Cov2,2); #covariances sum class 2 
    
    MeanAll=p1*Mean1 + p2*Mean2   #unbalanced classes

    #Spatial BETWEEN-CLASS MATRIX
    Sb=p1*(Mean1-MeanAll) @ np.transpose(Mean1-MeanAll) + p2*(Mean2-MeanAll) @ np.transpose(Mean2-MeanAll)  

    #Spatial WITHIN-CLASS MATRIX
    Sw=p1*Covavg1 + p2*Covavg2 
    Sw= (1-th)*Sw + th*np.eye(np.size(Sw,0),np.size(Sw,0))  
    
    V1, U1 = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
       
    
    rindices = np.argsort(-1*V1); 
    
    Vd1 = V1[rindices]               #ordered eigenvalues 
    #print(Vd1)
    V1 = np.diag(Vd1)                      
    U1 = U1[:,rindices]              #ordered eigenvectors (spatial filters)
    
    return U1, V1


"""
Spatial filter Projections
Gabriel Pires, June 2022 

"""

def FCB_projections(z1,z2,U):
    #initialize variables
    z1_f=np.zeros((np.size(z1,0),np.size(z1,1),np.size(z1,2)))
    z2_f=np.zeros((np.size(z2,0),np.size(z2,1),np.size(z2,2)))
  
    for i in np.arange(np.size(z1,2)):    #trials class 1
        z1_f[:,:,i] = np.transpose(U) @ np.squeeze(z1[:,:,i])

    for i in np.arange(np.size(z2,2)):    #trials class 2
        z2_f[:,:,i] = np.transpose(U) @ np.squeeze(z2[:,:,i])
    
    return z1_f, z2_f
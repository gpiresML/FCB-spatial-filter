# -*- coding: utf-8 -*-
"""
version Python 3.8.8
Created on Tue Jun  7 14:31:56 2022

RSQU   erg=rsqu(r, q) computes the r2-value for
       two one-dimensional distributions given by
       the vectors q and r

https://www.bci2000.org/mediawiki/index.php/Glossary#r-squared

r2 version adapted to Python
and r2 color map visualization

@author: Gabriel Pires, June 2022
"""
import numpy as np
#from matplotlib.colors import Normalize
#from matplotlib import cm
import matplotlib.pyplot as plt
import pylab

"""
r2 computation
"""
def rsquare(q,r):
    sum1 = np.sum(q);
    sum2 = np.sum(r);
    n1=np.size(q, axis=0);
    n2=np.size(r, axis=0);
    sumsqu1=np.sum(np.multiply(q,q));
    sumsqu2=np.sum(np.multiply(r,r));

    G=((sum1+sum2)**2)/(n1+n2);

    erg=(sum1**2/n1+sum2**2/n2-G)/(sumsqu1+sumsqu2-G);
    return erg


"""
Color map to visualize r2
"""
def plot_rsquare(t,ressq):
    data2plot = np.transpose(ressq)
    
    tamx=np.shape(data2plot)
    #print(tamx[0])  
    data2plot = np.concatenate( (data2plot, np.zeros((tamx[0],1)) ), 1)
    tamx=np.shape(data2plot)

    #print(np.shape(data2plot))  
    data2plot = np.concatenate( (data2plot, np.zeros((1,tamx[1])) ), 0)
    xData=t;
    xData=np.append(xData, xData[-1] + np.diff(xData[len(xData)-2 : len(xData)]));

    Nch=np.size(ressq, axis=1)

    #ax.pcolormesh(xData,np.arange(Nch+1),data2plot, vmin=-0.5, vmax=1.0)
    # ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)
    pylab.pcolor(xData, np.arange(Nch+1), data2plot, cmap=plt.cm.jet )
    pylab.colorbar()
    pylab.ylabel('Channels')
    pylab.xlabel('time (s)')
    pylab.title('Statistical r^2 between class1 and class2')
    pylab.show() 
    
    return 0
    

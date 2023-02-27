import math
import torch
import torch.nn as nn
import random
import scipy
import scipy.cluster
from scipy import stats
import numpy as np
from numpy import array
import matplotlib.pyplot as plot

fichdatos=open('casas.trn','r')
datos= [[float(cada) for cada in linea.strip().split()] for linea in fichdatos.readlines()]
numentradas=len(datos[0])-1

nd=[]
nuevo=open('casas2.trn','w')

for d in datos :
    nd.append(d[0]  + 2* random.random() -1)
    nd.append(d[1]  + 10*   random.random() -5)
    nd.append(d[2]+ 10*   random.random())
    if (random.random() >0.9):
        if d[3]==1: nd.append(0) 
        else: nd.append(1)
    else: nd.append(d[3]) 
    nd.append(d[4]  + 0.1*  random.random() -0.05)
    nd.append(d[5]  + 1*    random.random() -0.5)
    nd.append(d[6]  + 5*    random.random() -2.5)
    nd.append(d[7]  + 2*    random.random() -1)
    nd.append(d[8]  + 1.5*  random.random() -0.75)
    nd.append(d[9]  + 1*    random.random() -0.5)
    nd.append(d[10]+ 3*    random.random() -1.5)
    nd.append(d[11]+ 10*   random.random())
    nd.append(d[12]+ 4*    random.random() -2)
    nd.append(d[13]+ 3*    random.random() -1.5)
    nuevo.writelines(format(d))
    nuevo.writelines(format(nd))
    format(d)
    format(nd)
    nd=[]

nuevo.close()    


    
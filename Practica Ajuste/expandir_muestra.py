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
nuevo=open('casas_expandido.trn','w')

for d in datos :
    #Hay valores puntuales muy pequeños
    if(d[0]<0.1):
        nd.append(d[0]  +   0.01*random.random()    -0.005)
    else:
        nd.append(d[0]  +   0.1*random.random()    -0.05)
    if (d[1]==0):
        if (random.random() >0.8):
            nd.append(d[1]  +   2*random.random())
        else: 
            nd.append(0)    
    else:
        nd.append(d[1]  +   4*random.random()-2)

    nd.append(d[2]  +   2*random.random()-1)

    if (random.random() >0.9):
        if d[3]==1: nd.append(0) 
        else:       nd.append(1)
    else: nd.append(d[3])

    nd.append(d[4]  +   0.2*random.random()-0.1)
    nd.append(d[5]  +   1*random.random()-0.5)

    if(d[6]==100):
        nd.append(d[6]  -   10*random.random())
    else:
        if d[6]>90: 
            nd.append(d[6]  -   2*random.random()+1)
        else:  nd.append(d[6]  +   10*random.random()-5)
    nd.append(d[7]  +   2*random.random()-1)
    nd.append(d[8]  +   (int) (1.5*random.random()-0.75))
    nd.append(d[9]  +    20*random.random()-10)
    nd.append(d[10] +    10*random.random()-5)
    nd.append(d[11] +    40*random.random()-20)
    nd.append(d[12] +    10*random.random()-5)
    nd.append(d[13] +    8*random.random()-4)


    #Escribe la linea original
    for i in d:
        nuevo.write("{}\t".format("{:e}".format(i)))

    nuevo.write('\n')

    #A continuación, escribe la linea modificada aleatoriamente
    for j in nd:
        nuevo.write("{}\t".format("{:e}".format(j)))
        
    nuevo.write('\n')

    nd.clear()

nuevo.close()
fichdatos.close()    


    
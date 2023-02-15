#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Perceptrón para estimar precio de viviendas. Una capa oculta con 5 procesadores con activación tangente hiperbólica y error cuadrático.
"""

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
import graf0 as g

def septorch(datos,tipo,donde):
  entradas=[ [col for col in fila[:-1]] for fila in datos]
  salidas=[ fila[-1:] for fila in datos]
  redent=torch.tensor(entradas,dtype=tipo,device=donde,requires_grad=True)
  redsal=torch.tensor(salidas,dtype=tipo,device=donde)
  return redent,redsal,entradas,salidas

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") 

#Cargamos los datos, todos
fichdatos=open('casas.trn','r')
datos= [[float(cada) for cada in linea.strip().split()] for linea in fichdatos.readlines()]
numentradas=len(datos[0])-1

#Separamos ajuste y prueba
porazar=0.4
numuestras=len(datos)
muesajuste=0.7
muesval=0.15
#Desordena conjunto
random.shuffle(datos)
#Llevarlos a escala unitaria
datot=stats.zscore(array(datos,dtype=np.float32))
#Separa una parte para escoger por azar
limazar=int(porazar*numuestras)
datazar=datot[:limazar,:]
datgrup=datot[limazar:,:]

#Separa un primer lote de ajuste y prueba por azar
limajaz=int(limazar*muesajuste)
limvalaz=int(limazar*(muesajuste+muesval))
datajaz=datazar[:limajaz,:]
datvalaz=datazar[limajaz:limvalaz,:]
datpruaz=datazar[limvalaz:,:]
#Separa un segundo lote de ajuste y prueba por agrupamiento
limgrupaj=len(datgrup)
numajgrup=int(limgrupaj*muesajuste)
centros,grupos=scipy.cluster.vq.kmeans2(datgrup,numajgrup,minit='points')
orgpuntos=scipy.spatial.KDTree(datgrup)
dist,ind=orgpuntos.query(centros)
datajgrup=datgrup[ind]
indvalpru=np.setdiff1d(range(limgrupaj),ind)
datvalprugrup=datgrup[indvalpru]
numprugrup=int(limgrupaj*(1-muesval-muesajuste))
centros,grupos=scipy.cluster.vq.kmeans2(datvalprugrup,numprugrup,minit='points')
orgpuntos=scipy.spatial.KDTree(datvalprugrup)
dist,ind=orgpuntos.query(centros)
datprugrup=datvalprugrup[ind]
indpru=np.setdiff1d(range(len(datvalprugrup)),ind)
datvalgrup=datvalprugrup[indpru]

dataj=np.vstack((datajaz,datajgrup))
datval=np.vstack((datvalaz,datvalgrup))
datpru=np.vstack((datpruaz,datprugrup))

#Pasarlo a tensores torch
tea,tsa,ea,sa=septorch(dataj,dtype,device)
tev,tsv,ev,sv=septorch(datval,dtype,device)
tep,tsp,ep,sp=septorch(datpru,dtype,device)

#Si interesa la referencia de regresión lineal, Torch no la tiene directamente, pero puedes hacer una red enteramente lineal y ajustarla


#definir red
ocultos=5
red=nn.Sequential(
    nn.Linear(numentradas, ocultos),
    nn.Tanh(),#Para un modelo lineal, suprimirías esto
    nn.Linear(ocultos, 1),
)
#).cuda(device)
#definir error a optimizar
error = nn.MSELoss()
#definir algoritmo de ajuste
ajuste=torch.optim.LBFGS(red.parameters(),lr=0.01,max_iter=50,history_size=10)

nvalfal=0
def evalua():
        ajuste.zero_grad()
        s = red(tea)
        e = error(s, tsa)
        e.backward()
        return e
print ("Iteración","Error de ajuste","Error de validación")
for it in range(100): # Calcula salidas 
  ea=evalua()
  salval = red(tev)
  ev=error(salval,tsv)
  if 'evprevio' in dir():
    if evprevio<ev.item():
      nvalfal=nvalfal+1
    else:
      nvalfal=0
  if nvalfal>5:
    break
  evprevio=ev.item()
  print(it, math.sqrt(ea.item()),math.sqrt(evprevio))

  ajuste.step(evalua)

#Prueba: pasada directa, incluyendo derivadas de la red
ajuste.zero_grad()
salpru=red(tep)
ep=error(salpru,tsp)
print("Error de prueba",math.sqrt(ep.item()))

"""
numvar1=1
numvar2=2

conj,tablagraf=g.grafini(numvar1,numvar2)
for var1 in range(numvar1):
  for var2 in range(numvar2):
    g.grafindiv(tablagraf,var1,var2,datos[:,var1], datos[:,var2])
g.grafconc(conj,"Vars2 respecto a vars1")
"""
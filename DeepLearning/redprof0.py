#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Perceptrón para estimar precio de viviendas. 
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

def septorch(datos,tipo,donde):
  entradas=datos[:,:-1]
  salidas=datos[:,-1:]
  redent=torch.tensor(entradas,dtype=tipo,device=donde)
  redsal=torch.tensor(salidas,dtype=tipo,device=donde)
  return redent,redsal

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") 

#Cargamos los datos, todos
fichdatos=open('casas.trn','r')
datos_sin= [[float(cada) for cada in linea.strip().split()] for linea in fichdatos.readlines()]
numentradas=len(datos_sin[0])-1
datos=stats.zscore(array(datos_sin,dtype=np.float32))

#Separamos ajuste y prueba
porazar=0.6
numuestras=len(datos)
muesajuste=0.3
#Desordena conjunto
random.shuffle(datos)
#Separa una parte para escoger por azar
limazar=int(porazar*numuestras)
datazar=datos[:limazar]
datgrup=datos[limazar:]

#Separa un primer lote de ajuste y prueba por azar
limajaz=int(limazar*muesajuste)
datajaz=datazar[:limajaz]
datpruaz=datazar[limajaz:]
#Separa un segundo lote de ajuste y prueba por agrupamiento
datkm=array(datgrup,dtype=np.float32)
limgrupaj=len(datgrup)
numajgrup=int(limgrupaj*muesajuste)
centros,grupos=scipy.cluster.vq.kmeans2(datkm,numajgrup,minit='points')
orgpuntos=scipy.spatial.KDTree(datkm)
dist,ind=orgpuntos.query(centros)
datajgrup=datkm[ind]
indvalpru=np.setdiff1d(range(limgrupaj),ind)
datprugrup=datkm[indvalpru]

dataj=np.vstack((array(datajaz,dtype=np.float32),datajgrup))
datpru=np.vstack((array(datpruaz,dtype=np.float32),datprugrup))

#Pasarlo a tensores torch
tea,tsa=septorch(dataj,dtype,device)
tep,tsp=septorch(datpru,dtype,device)



#definir red

a=150
b=200
c=90
red=nn.Sequential(
    nn.Linear(numentradas,a),
    nn.Tanh(),
    nn.Linear(a, b),
    #nn.Dropout(p=0.2),
    nn.Tanh(),
    nn.Linear(b,c),
    nn.Dropout(p=0.3),
    nn.Tanh(),
    nn.Linear(c, 1),
    #nn.Dropout(p=0.1),
)
#Lo anterior también se podría hacer
#red=nn.Sequential(
#    nn.LazyLinear(a),
#    nn.Tanh(),
#    nn.LazyLinear( b),
#    nn.Tanh(),
#    nn.LazyLinear(c),
#    nn.Tanh(),
#    nn.LazyLinear(1),
#)
#pero requiere una inicialización, pasando algún dato de cara, por ejemplo
#s=red(tea)
#).cuda(device)

#definir error a optimizar

error = nn.SmoothL1Loss()
var_lr=0.01
#definir algoritmo de ajuste
#ajuste=torch.optim.LBFGS(red.parameters(),lr=var_lr,max_iter=50,history_size=10) #Cuasi-newton
#ajuste=torch.optim.Rprop(red.parameters(), lr=var_lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50)) # Siguiendo gradiente a paso fijo (factores de aumento y disminución y pasos mínimo y máximo)
#ajuste=torch.optim.SGD(red.parameters(),lr=var_lr, weight_decay=100) #Descenso gradiente estocástico (o gradiente puro si momento = 0)ç
#ajuste=torch.optim.Adam(red.parameters(), lr=var_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) # Gradiente estocástico dividido por módulo, betas son momentos

ajuste=torch.optim.Adam(red.parameters(),lr=var_lr,weight_decay=0.05) # descenso gradiente con corrección automática de módulo
#ajuste=torch.optim.RMSprop(red.parameters(),lr=var_lr, momentum=0, weight_decay=0.0001) #descenso gradiente sólo con el signo

nminibatch=150
indicaj=np.array([0,nminibatch])
for it in range(100): 
 while indicaj[0]<len(tea):
  def evalua():
        ajuste.zero_grad()
        s = red(tea[indicaj[0]:indicaj[1],:])
        e = error(s, tsa[indicaj[0]:indicaj[1],:])
        e.backward()
        return e
  ea=evalua()
  #print(it, math.sqrt(ea.item())) 
  ajuste.step(evalua)
  indicaj=indicaj+nminibatch
  if indicaj[1]>len(tea):
    indicaj[1]=len(tea)
 indicaj=np.array([0,nminibatch])
 """
def evalua():
        ajuste.zero_grad()
        s = red(tea)
        e = error(s, tsa)
        e.backward()
        return e
for it in range(100): # Calcula salidas 
  ea=evalua()
  print(it, math.sqrt(ea.item())) #Si no usas el cuadrático quita la raíz

  ajuste.step(evalua)

"""

#Prueba
salpru=red(tep)
ep=error(salpru,tsp)
print(math.sqrt(ep.item())) #Si no usas el cuadrático quita la raíz

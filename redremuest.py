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
datos__s= [[float(cada) for cada in linea.strip().split()] for linea in fichdatos.readlines()]
datos=stats.zscore(array(datos__s,dtype=np.float32))
numentradas=len(datos[0])-1

#Separamos ajuste y prueba
porazar=0.6
numuestras=len(datos)
muesajuste=0.3
muesval=0.1

numestim= 100 #Numero de particiones de la red grande
limaj=np.zeros(numestim)
difer=np.zeros(numestim)
for v in range(numestim):
  #Desordena conjunto
  random.shuffle(datos)
  #Separa una parte para escoger por azar
  limazar=int(porazar*numuestras)
  datazar=datos[:limazar]
  datgrup=datos[limazar:]

  #Separa un primer lote de ajuste y prueba por azar
  limajaz=int(limazar*muesajuste)
  limvalaz=int(limazar*(muesajuste+muesval))
  datajaz=datazar[:limajaz]
  datvalaz=datazar[limajaz:limvalaz]
  datpruaz=datazar[limvalaz:]
  #Separa un segundo lote de ajuste y prueba por agrupamiento
  datkm=array(datgrup,dtype=np.float32)
  limgrupaj=len(datgrup)
  numajgrup=int(limgrupaj*muesajuste)
  centros,grupos=scipy.cluster.vq.kmeans2(datkm,numajgrup,minit='points')
  orgpuntos=scipy.spatial.KDTree(datkm)
  dist,ind=orgpuntos.query(centros)
  datajgrup=datkm[ind]
  indvalpru=np.setdiff1d(range(limgrupaj),ind)
  datvalprugrup=datkm[indvalpru]
  numprugrup=int(limgrupaj*(1-muesval-muesajuste))
  centros,grupos=scipy.cluster.vq.kmeans2(datvalprugrup,numprugrup,minit='points')
  orgpuntos=scipy.spatial.KDTree(datvalprugrup)
  dist,ind=orgpuntos.query(centros)
  datprugrup=datvalprugrup[ind]
  indpru=np.setdiff1d(range(len(datvalprugrup)),ind)
  datvalgrup=datvalprugrup[indpru]

  dataj=np.vstack((array(datajaz,dtype=np.float32),datajgrup))
  datval=np.vstack((array(datvalaz,dtype=np.float32),datvalgrup))
  datpru=np.vstack((array(datpruaz,dtype=np.float32),datprugrup))

  #Pasarlo a tensores torch
  tea,tsa=septorch(dataj,dtype,device)
  tev,tsv=septorch(datval,dtype,device)
  tep,tsp=septorch(datpru,dtype,device)

  #definir red RED GRANDE
  ocultos=100
  red=nn.Sequential(
      nn.Linear(numentradas, ocultos),
      nn.Tanh(),
      nn.Linear(ocultos, 1),
  )
  #).cuda(device)
  #definir error a optimizar
  error = nn.SmoothL1Loss()

  #definir algoritmo de ajuste
  ajuste=torch.optim.Rprop(red.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50)) # Siguiendo gradiente a paso fijo (factores de aumento y disminución y pasos mínimo y máximo)

  nvalfal=0
  def evalua():
          ajuste.zero_grad()
          s = red(tea)
          e = error(s, tsa)
          e.backward()
          return e
  for it in range(100): 
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

    ajuste.step(evalua)

  limaj[v]=ea.item() #AQUI GUARDA DONDE PARA EL AJUSTE
    #Prueba
  salpru=red(tep)
  ep=error(salpru,tsp)
  print(it,math.sqrt(ep.item())) #Si no usas el cuadrático quita la raíz
  difer[v]=ep.item()-ea.item() #GUARDA LA DIFERENCIA ENTRE LA PRUEBA Y EL AJUSTE, CUANTO HA CAMBIADO


  #HASTA AQUI LLEGA LO DE LA RED GRANDE, CUANDO TERMINA TIENE LIMAJ Y DIFER
  
tea,tsa=septorch(array(datos,dtype=np.float32),dtype,device) #A CONTINUACION AGARRA TODA LA MUESTRA
#definir red
ocultos=25
red=nn.Sequential(
    nn.Linear(numentradas, ocultos),
    nn.Tanh(),
    nn.Linear(ocultos, 1),
)
#).cuda(device)
#definir error a optimizar
error = nn.SmoothL1Loss()

#definir algoritmo de ajuste
ajuste=torch.optim.Rprop(red.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50)) # Siguiendo gradiente a paso fijo (factores de aumento y disminución y pasos mínimo y máximo)


#AQUI NO HAY VALIDACION EN LA FASE 2 TODO ES CONJUNTO DE AJUSTE
def evalua():
        ajuste.zero_grad()
        s = red(tea)
        e = error(s, tsa)
        e.backward()
        return e
tope=np.median(limaj)# o mean o .... LO QUE SEA
for it in range(100): 
  ea=evalua()
  if ea.item()<tope: #CUANDO LLEGA AL TOPE SE SALE
    break
  print(it, math.sqrt(ea.item())) #Si no usas el cuadrático quita la raíz

  ajuste.step(evalua)

print(math.sqrt(ea.item()+np.median(difer))) #Si no usas el cuadrático quita la raíz


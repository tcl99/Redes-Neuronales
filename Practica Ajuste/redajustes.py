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
fichdatos=open('casas_expandido.trn','r')
datos= [[float(cada) for cada in linea.strip().split()] for linea in fichdatos.readlines()]
numentradas=len(datos[0])-1

#Separamos ajuste y prueba

porazar=0.6           #VALOR AJUSTABLE
numuestras=len(datos)
muesajuste=0.3        #VALOR AJUSTABLE
muesval=0.1          #VALOR AJUSTABLE
#Desordena conjunto
random.shuffle(datos)

#NUEVO
#LLEVARLO A ESCALA UNITARIA
#NORMALIZAR
datos_norm=stats.zscore(array(datos, dtype=np.float32))
#Separa una parte para escoger por azar
limazar=int(porazar*numuestras)
#datazar=datos[:limazar]
#datgrup=datos[limazar:]
datazar=datos_norm[:limazar]
datgrup=datos_norm[limazar:]

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

#definir red
ocultos=25
red=nn.Sequential(
    nn.Linear(numentradas, ocultos),
    nn.Tanh(),
    nn.Linear(ocultos, 1),
)
#).cuda(device)
#definir error a optimizar


class ErrorSatura(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, red, reales):
        error2=(red-reales)**2    #ERROR CUADRATICO
        satura=torch.log(1+error2)#ERROR LOGARITMICO 1
        sumcuad=torch.mean(satura)#ERROR QUE SE LE DIGA
        return sumcuad
    
def presensatura(error):
    return math.sqrt(math.exp(error)-1)

#error=ErrorSatura()       #en torno a 3
#error = nn.MSELoss()       #la peor, no baja de 5
#error = nn.L1Loss()        # en torno a 2.5
error = nn.SmoothL1Loss()   # en torno a 2

presenerror=math.sqrt
#presenerror=presensatura


#definir algoritmo de ajuste
#5 PASO
#NOS IMPORTA 3 COSAS
#OPTIMO(ERROR MINIMO)       TIEMPO(Q TARDA EN EJECUTARSE)        ESTABILIDAD(VALORES QUE DA CAMBIAN MUCHO O NO)
#0,34                           rapido pero mas lento               0,34 y 0,40
#0.270                         RAPIDISIMO                           0,27 Y 0,32
#0.38                                                               Depende del momentum
#0.272                          RAPIDISIMO                          0,27 Y 0,33 (0,5)
#0.272                          RAPIDISMO                           0,28 Y 0,38 tira pa altos

var_lr=0.01
#ajuste=torch.optim.LBFGS(red.parameters(),lr=var_lr,max_iter=50,history_size=10) #Cuasi-newton
#ajuste=torch.optim.LBFGS(red.parameters(),lr=var_lr,tolerance_grad=1e-03, tolerance_change=1e-03, max_iter=50,history_size=10) #Más parámetros en cuasinewton
#ajuste=torch.optim.SGD(red.parameters(),lr=var_lr,momentum=0.05,dampening=0.1) #Descenso gradiente estocástico (o gradiente puro si momento = 0)ç
ajuste=torch.optim.Rprop(red.parameters(), lr=var_lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50)) # Siguiendo gradiente a paso fijo (factores de aumento y disminución y pasos mínimo y máximo)
#ajuste=torch.optim.Adam(red.parameters(), lr=var_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) # Gradiente estocástico dividido por módulo, betas son momentos

#SEGUN LA MUESTRA, EL RENDIMIENTO DE LOS ALGORITMOS VARÍA
#cuasi newton lleva mal las muestras grandes, mejor para medianos
#paso fijo y aproximacion modulo son variaciones del gradiente estocastico para intentar mejorar, para problemas grandes suelen ser mejor estos 2
#si se baja el learning rate, baja el tiempo pero se pierde estabilidad

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
  print(it, presenerror(ea.item()),presenerror(evprevio)) #Recuerda actualizar presenerror cuando cambies de medida de error

  ajuste.step(evalua)

#Prueba
salpru=red(tep)
ep=error(salpru,tsp)
print(presenerror(ep.item())) #Actualizar presenerror
print(datos_norm.std(axis=0))



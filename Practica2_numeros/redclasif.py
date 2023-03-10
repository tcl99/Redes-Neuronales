#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Perceptrón para reconocer imágenes de cifras. 
"""

import math
import numpy as np
import torch
import torch.nn as nn
import random
import scipy.io as sio

lrate=0.02
capaocultos=15
ajuste_val=0.7
fallosval=5
#errorval
#problemas


def septorch(datos,tipo,donde):
  entradas=datos[:,:-10]
  salidas=datos[:,-10:]
  redent=torch.tensor(entradas,dtype=tipo,device=donde)
  redsal=torch.tensor(salidas,dtype=tipo,device=donde)
  return redent,redsal

#Tipos y hardware a usar
dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") 

#Cargamos los datos, todos
matrices=sio.loadmat('mnist_uint8.mat')

#Separamos ajuste y validacion
datos=np.hstack((matrices['train_x'],matrices['train_y'])).astype(float)
datpru=np.hstack((matrices['test_x'],matrices['test_y'])).astype(float)
del matrices
numvarent=len(datos[0])-10
numuestras=len(datos)
#La prueba ya está separada. Partimos sólo ajuste y validación
muesajuste=ajuste_val#Fracción de muestra que se usa para ajuste
#Desordena conjunto
random.shuffle(datos)
#Los agrupamientos y búsqueda de vecinos próximos son muy problemáticos con muchos puntos y muchas dimensiones
#Separamos todo por azar
limajaz=int(numuestras*muesajuste)
dataj=datos[:limajaz]
datval=datos[limajaz:]
del datos

def clases(matriz):
	valor,clase=torch.max(matriz,1)
	return clase
#Pasarlo a tensores torch
tea,tsa=septorch(dataj,dtype,device)
tev,tsv=septorch(datval,dtype,device)
tep,tsp=septorch(datpru,dtype,device)

#Si interesa la referencia de regresión lineal, Torch no la tiene directamente, pero puedes hacer una red enteramente lineal y ajustarla


#definir red
ocultos=capaocultos#¿Cuántos en la capa oculta?
red=nn.Sequential(
    nn.Linear(numvarent, ocultos),
    nn.Tanh(),
    nn.Linear(ocultos, 10),
    nn.Sigmoid(), #Capa de salida para que esté entre 0 y 1, porcentaje de probabilidad
)
#).cuda(device)
#definir error a optimiza
error = nn.MSELoss() 
#Genera una matriz de ocnfusión
#Matriz que relaciona los aciertos y errores de la red
def matconf(matsal,matreal):
	clasesal=clases(matsal)
	clasereal=clases(matreal)
	numclases=len(matsal[0])
	numcasos=len(clasereal)
	relacion=[[0 for col in range(numclases)] for fil in range(numclases)]
	for caso in range(numcasos):
		relacion[clasereal[caso]][clasesal[caso]]=relacion[clasereal[caso]][clasesal[caso]]+1
	return relacion
	
def errorclas(matsal,matreal):
	clasesal=clases(matsal)
	clasereal=clases(matreal)
	error=[(clasesal[caso]!=clasereal[caso]).item() for caso in range(len(clasereal))]
	return sum(error)/float(len(clasereal))
	
#definir algoritmo de ajuste, opmitizador, learning rate
ajuste=torch.optim.LBFGS(red.parameters(),lr=lrate,max_iter=50,history_size=10)# cuidado con lr grande, es fácilmente inestable

nvalfal=0
def evalua():
        ajuste.zero_grad()
        s = red(tea)
        e = error(s, tsa)
        e.backward()
        return e
for it in range(50): #más iteraciones supone más tiempo. Indica el limite de iteraciones
  ea=evalua()
  salval = red(tev)
  ev=error(salval,tsv)
  if 'evprevio' in dir():
    if evprevio<ev.item():
      nvalfal=nvalfal+1
    else:
      nvalfal=0
  if nvalfal>fallosval: #¿Cuántos fallos de validación seguidos?
    break
  evprevio=ev.item()
  print(it, ea.item(),evprevio,errorclas(salval,tsv))

  ajuste.step(evalua)

#Prueba
salpru=red(tep)
ep=error(salpru,tsp)
print(ep.item(),errorclas(salpru,tsp))
print (matconf(salpru,tsp))

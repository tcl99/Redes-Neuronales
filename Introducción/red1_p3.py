#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
PerceptrÃ³n para estimar precio de viviendas. 
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

var_muesajuste=0.6
var_muesval=0.2
#Recordar q hay3 grupos, si suma 1 queda 0 para el de comprobar
var_ocultos=15
var_error=nn.MSELoss()
var_lr=0.005
var_nvalfal=5


def septorch(datos,tipo,donde):
  entradas=[ [col for col in fila[:-1]] for fila in datos]
  salidas=[ fila[-1:] for fila in datos]
  redent=torch.tensor(entradas,dtype=tipo,device=donde,requires_grad=True)
  redsal=torch.tensor(salidas,dtype=tipo,device=donde)
  return redent,redsal,entradas,salidas

def grafini(numfils,numcols):
  return plot.subplots(numfils, numcols, sharex='col', sharey='row', figsize=(20, 15))

def grafhintini(numcapas):
  return plot.subplots(numcapas, 1, sharex='col', sharey='row', figsize=(20, 15))

def grafindiv(figtot,figind,f,c,x,y):
      figind[f,c].scatter(x, y)
      figind[f,c].minorticks_off()
      figind[f,c].locator_params(tight=True, nbins=4)

def grafhintcada(pesos):
  fig,cada=plot.subplots(len(pesos), 1, sharex='col', sharey='row', figsize=(20, 15))
  if len(pesos)>1:
    for filaproc in range(len(pesos)):
      cada[filaproc].bar(range(pesos.shape[1]), pesos[filaproc,:])
  else:
    cada.bar(range(pesos.shape[1]), pesos[0,:])
  plot.suptitle("Valores de los pesos")
  plot.show()

def numhintcada(pesos):
  valpesos=np.absolute(pesos)
  maxpeso=np.maximum.reduce(pesos,axis=None)
  gordos=valpesos>0.8*maxpeso
  print ("Pesos gordos (procesador,variable,valor): ",gordos.nonzero(),pesos[gordos])

def grafconc(x,rotulo):
  plot.suptitle(rotulo)
  plot.show()
  
def procesacapas(red,previo,sini):
  salcapa=[]
  guardar=sini#La primera es la salida de lo lineal, que no interesa. Por ese patrÃ³n, irÃ¡n alternando. 
  for capa in red.children():
    previo=capa(previo)
    if guardar:
      salcapa.append(previo)
    guardar=not guardar
  return salcapa
  
dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") 

#Cargamos los datos, todos
fichdatos=open('casas.trn','r')
datos= [[float(cada) for cada in linea.strip().split()] for linea in fichdatos.readlines()]
numentradas=len(datos[0])-1

#Separamos ajuste y prueba
numuestras=len(datos)
muesajuste=var_muesajuste
muesval=var_muesval
#Desordena conjunto
random.shuffle(datos)
#Llevarlos a escala unitaria
datot=stats.zscore(array(datos,dtype=np.float32))

#Separa un primer lote de ajuste y prueba por azar
limajaz=int(numuestras*muesajuste)
limvalaz=int(numuestras*(muesajuste+muesval))
dataj=datot[:limajaz,:]
datval=datot[limajaz:limvalaz,:]
datpru=datot[limvalaz:,:]

#Pasarlo a tensores torch
tea,tsa,ea,sa=septorch(dataj,dtype,device)
tev,tsv,ev,sv=septorch(datval,dtype,device)
tep,tsp,ep,sp=septorch(datpru,dtype,device)

#Si interesa la referencia de regresiÃ³n lineal, Torch no la tiene directamente, pero puedes hacer una red enteramente lineal y ajustarla


#definir red
ocultos=var_ocultos
red=nn.Sequential(
    nn.Linear(numentradas, ocultos),
    nn.Tanh(),#Para un modelo lineal, suprimirÃ­as esto. TambiÃ©n puedes poner otras funciones (HardTanh, Sigmoid, ReLU)
    nn.Linear(ocultos, 1),
)
#).cuda(device)
#definir error a optimizar
error = var_error #MSELoss L1Loss SmoothL1Loss nn.()
#definir algoritmo de ajuste
ajuste=torch.optim.LBFGS(red.parameters(),lr=var_lr,max_iter=50,history_size=10)

nvalfal=0
def evalua():
        ajuste.zero_grad()
        s = red(tea)
        e = error(s, tsa)
        e.backward()
        return e
print ("IteraciÃ³n","Error de ajuste","Error de validaciÃ³n")
for it in range(100): # Calcula salidas 
  ea=evalua()
  salval = red(tev)
  ev=error(salval,tsv)
  if 'evprevio' in dir():
    if evprevio<ev.item():
      nvalfal=nvalfal+1
    else:
      nvalfal=0
  if nvalfal>var_nvalfal:
    break
  evprevio=ev.item()
  print(it, math.sqrt(ea.item()),math.sqrt(evprevio))

  ajuste.step(evalua)

#Prueba: pasada directa, incluyendo derivadas de la red
ajuste.zero_grad()
salpru=red(tep)
ep=error(salpru,tsp)
print("Error de prueba",math.sqrt(ep.item()))

###AnÃ¡lisis adicionales
##Derivadas respecto a las entradas
for salida in salpru:
  salida.backward(retain_graph=True)
funanal={'pesini':grafini,'pescada':grafindiv,'pesfin':grafconc,'hinton':grafhintcada}
numvartot=tep.shape[1]
#Sacamos en pantalla las medias y las desviaciones tÃ­picas
dermed=tep.grad.numpy().mean(axis=0)
print ("Derivadas medias",dermed)
print ("DesviaciÃ³n tÃ­pica de derivadas",tep.grad.numpy().std(axis=0))
sal1,sal2=funanal['pesini'](numvartot,numvartot+1)
for vardev in range(numvartot):
  for var in range(numvartot):
    funanal['pescada'](sal1,sal2,vardev,var,tep[:,var].detach().numpy(), tep.grad[:,vardev].detach().numpy())
  funanal['pescada'](sal1,sal2,vardev,numvartot,tsp.detach().numpy().squeeze(), tep.grad[:,vardev].detach().numpy())
funanal['pesfin'](sal1,"Derivadas respecto a entradas frente a entradas")
##Actividad de procesadores ocultos:
#pasada capa a capa
previo=tep
salcapa=procesacapas(red,previo,False)
#salcapa.pop() Si hubiera que quitar la de salida
#Graficarlo respecto a variables si es manejable. Si no, simplemente sacar las dependencias procesador/variable mÃ¡s significativas
for capa in salcapa:
  #Sacar varianza y mostrar los que la tengan pequeÃ±a
  varian=capa.var()
  inutiles=varian<0.15
  if len(varian[inutiles])>0:
    print ("Procesadores poco activos: ",inutiles.nonzero())
  numprocs=capa.shape[1]
  sal1,sal2=funanal['pesini'](numprocs,numvartot+1)
  for proc in range(numprocs):
    for var in range(numvartot):
      funanal['pescada'](sal1,sal2,proc,var,tep[:,var].detach().numpy(), capa[:,proc].detach().numpy())
    funanal['pescada'](sal1,sal2,proc,numvartot,tsp.detach().numpy().squeeze(), capa[:,proc].detach().numpy())
  funanal['pesfin'](sal1,"Procesadores frente a variables")
##Mostrar pesos. Cada procesador un diagrama de barras respecto a las variables. Si son muchos, indicar sÃ³lo los mÃ¡s significativos
tienepesos=True#La primera es la salida de lo lineal, que tiene pesos. Por ese patrÃ³n, irÃ¡n alternandoLa de funciÃ³n de activaciÃ³n, no. Si el patrÃ³n se mantiene, irÃ¡n alternando
for capa in red.children():
  if tienepesos:
    pesos=capa.parameters()#procesadores en filas, variables en columnas
    p= next(pesos)#Poner que sÃ³lo interesan capas salternas
    funanal['hinton'](p.detach().numpy())
  tienepesos=not tienepesos
#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import random
import scipy
import scipy.cluster
from scipy import stats
from numpy import array
import math
import matplotlib.pyplot as plot

def septorch(datos,tipo,donde):
  entradas=[ [col for col in fila[:-1]] for fila in datos]
  salidas=[ fila[-1:] for fila in datos]
  redent=torch.tensor(entradas,dtype=tipo,device=donde,requires_grad=True)
  redsal=torch.tensor(salidas,dtype=tipo,device=donde)
  return redent,redsal

class Datos(data.Dataset): #data.Dataset
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return (x, y)
 
tamsubmues=100
npruebas=20
var_error=nn.SmoothL1Loss()
var_lr=0.01


#Cargamos los datos, todos
fichdatos=open('casas.trn','r')
datos= [[float(cada) for cada in linea.strip().split()] for linea in fichdatos.readlines()]
numentradas=len(datos[0])-1
datot=stats.zscore(array(datos,dtype=np.float32))
dtype = torch.float
device = torch.device("cpu")
te,ts=septorch(datot,dtype,device)
total = Datos(te, ts)

#EN ESTA PARTE ES DONDE SACAS LA MUESTRA, EL TAMAÑO DE LA MUESTRA TOTAL SE ELIGE
#BOOTSTRAP
#selec = data.RandomSampler(total,num_samples=tamsubmues, replacement=True)
#muestras=data.DataLoader(total,batch_size=tamsubmues, sampler=selec)

#VALIDACIÓN CRUZADA
numcrosval= 10 #en cuantos trozos lo partes, lo tipico es 5, 10
muestras=data.DataLoader(total,batch_size=len(datos)//numcrosval, shuffle=True) #validación cruzada
#Si iteras sancado entradas y salidas y las metes en una lista, tendrás los conjuntos
#Para concatenar varios, puedes usar torch.vstack((parte1,parte2,...))
#torch.vstack(listaajuste)

#definir red
ocultos=25
red=nn.Sequential(
    nn.Linear(numentradas, ocultos),
    nn.Tanh(),
    nn.Linear(ocultos, 1),
)

#NPRUEBAS ES EL NUMERO DE VECES QUE SE "RETIRA LA BOLA DE LA BOLSA, EL NUMERO DE VECES Q COGES UNA PARTE DE LA MUESTRA"
rmse=np.zeros(npruebas) #npruebas: ¿Cuántas extracciones?
#ciclo para montar la lista de trozos
lista_trozos=[]
e_trozos=[]
s_trozos=[]
rmse=[]

for remues in muestras:
    ent, sal= remues
    e_trozos.append(ent)
    s_trozos.append(sal)

for i,trozo_prueba in enumerate(zip(e_trozos, s_trozos)): 
        #Ciclo de ajustar y probar la red
         e_resto_lista=e_trozos[:]
         e_resto_lista.pop(i)
         e_ajuste=torch.vstack(e_resto_lista)
         #salida
         s_resto_lista=s_trozos[:]
         s_resto_lista.pop(i)
         s_ajuste=torch.vstack(s_resto_lista)

         error = var_error #MSELoss L1Loss SmoothL1Loss nn.()
        #definir algoritmo de ajuste
         ajuste=torch.optim.LBFGS(red.parameters(),lr=var_lr,max_iter=50,history_size=10) #Cuasi-newton
        # ajuste=torch.optim.Rprop(red.parameters(), lr=var_lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50)) # Siguiendo gradiente a paso fijo (factores de aumento y disminución y pasos mínimo y máximo)
         nvalfal=0
         def evalua():
                    ajuste.zero_grad()
                    s = red(e_ajuste)
                    e = error(s, s_ajuste)
                    e.backward()
                    return e
         print ("IteraciÃ³n","Error de ajuste","Error de validaciÃ³n")
         for it in range(100): # Calcula salidas 
                ea=evalua()
                salval = red(e_ajuste)
                ev=error(salval,s_ajuste)
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
        
         #Prueba: 
         with torch.no_grad():
             #te,ts=trozo_prueba
             salpru=red(e_ajuste)
             ep=error(salpru,s_ajuste)
             rmse.append(math.sqrt(ep.item()))
plot.hist(rmse)#Si te llevas bien con matplotlib
plot.show()
print(rmse)


                

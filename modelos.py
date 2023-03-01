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

class Datos(Dataset): #data.Dataset
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return (x, y)
 

#Cargamos los datos, todos
fichdatos=open('casas.trn','r')
datos= [[float(cada) for cada in linea.strip().split()] for linea in fichdatos.readlines()]
numentradas=len(datos[0])-1
datot=stats.zscore(array(datos,dtype=np.float32))
dtype = torch.float
device = torch.device("cpu")
te,ts=septorch(datot,dtype,device)
total = Datos(te, ts)
selec = data.RandomSampler(total,num_samples=tamsubmues, replacement=True)
#EN ESTA PARTE ES DONDE SACAS LA MUESTRA, EL TAMAÑO DE LA MUESTRA TOTAL SE ELIGE
muestras=data.DataLoader(total,batch_size=tamsubmues, sampler=selec) 
#muestras=data.DataLoader(total,batch_size=numuestras//numcrosval, shuffle=True) validación cruzada
#Si iteras sancado entradas y salidas y las metes en una lista, tendrás los conjuntos
#Para concatenar varios, puedes usar torch.vstack((parte1,parte2,...))

#NPRUEBAS ES EL NUMERO DE VECES QUE SE "RETIRA LA BOLA DE LA BOLSA, EL NUMERO DE VECES Q COGES UNA PARTE DE LA MUESTRA"
rmse=np.zeros(npruebas) #npruebas: ¿Cuántas extracciones?
for p,remues in zip(range(npruebas),muestras): 
         ent_sub,sal_sub= remues
         #Aquí ajustas la red. Supongamos que se llama red AQUI VA LO DE ERROR=MSELOSS Y TAL

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
        
         #Prueba: 
         with torch.no_grad():
             salpru=red(te)
             ep=error(salpru,ts)
             rmse[p]=math.sqrt(ep.item())
plot.hist(rmse)#Si te llevas bien con matplotlib
plot.show()


                

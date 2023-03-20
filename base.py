#!/usr/bin/python3
# -*- coding: utf-8 -*-

import copy
from numpy import array
import random
import numpy as np
import torch
import torch.nn as nn
import PIL
from PIL import Image
import math
import scipy.io as sio

def clases(matriz):
	clase=matriz.argmax(axis=1)
	return clase

def torchea(entradas,salidas,tipo,donde):
  """
Recibe entradas y salidas, pasándolo a torch, al dispositivo indicado
  """
  redent=torch.tensor(entradas,dtype=tipo,device=donde)
  redsal=torch.tensor(salidas,dtype=tipo,device=donde)
  return redent,redsal

def errorclas(clasesal, clasereal): #Calcula el error en tanto por uno
    error = [clasesal[caso,:].argmax() != clasereal[caso,:].argmax() for caso in range(len(clasereal))]
    return sum(error) / float(len(clasereal))

def menser(error):
    """
Sacar la precisión a partir del error (complementario)
>>>menser(0.1)
Aciertos=0.9
    """     
    return "Aciertos={}".format(1-error)

def mostrar(clasred,salidas):
              """
Mostrar un caso al azar: imagen de entrada, salida ideal y salida de la red
              """     
              global datpru
              nmax=len(datpru)-1
              i1=random.randint(0,nmax)
              imagorig=Image.fromarray(datpru[i1:i1+1,0,:,:].squeeze()*255)
              imagorig.resize((200,200),PIL.Image.BILINEAR).show()
              print("Real: {}".format(salidas[i1]),'Red: {}'.format(clasred[i1]))

def ejemplos(red):
   """
Control del ciclo de ir mostrando ejemplos con una red
   """
   global tep,salpru
   seguir="s"
   salidas=clases(salpru)
   salred=red(tep.cpu())
   clasred=clases(salred.detach().numpy())
   while(seguir=="s"):
    mostrar(clasred,salidas) 
    seguir=input("Teclea s si quierés más ")
	  
def optiauxpar(red0, limajaz, inda, indv):
    """
   Pasada de optimización
Recibe una red, la cantidad de casos de ajuste, sus índices, los de validación y los de prueba
Devuelve el error final y la red ajustada
    """
    global datos,salidas,tep,tsp
    global device, error, ajuste, ajustepaso
    global it,evprevio, args
    #Pasarlo a tensores torch
    tea,tsa=torchea(datos[inda,:,:,:],salidas[inda,:],dtype,device)
    tev,tsv=torchea(datos[indv,:,:,:],salidas[indv,:],dtype,device)
    if proc=="cpu":
        red = copy.deepcopy(red0)
    else:
        red = copy.deepcopy(red0).cuda(device)
    limalg=args.limit
    ajuste=algajuste(red.parameters(), lr=args.learningrate, weight_decay=args.weightdecay)
    try:
            evprevio=1e9
            nvalfal = 0
            numcasos=args.minibatch
            for it in range(limalg):
                numcasos = min(numcasos, limajaz)
                numbloques = limajaz // numcasos
                limites = [(cual * numcasos, (cual + 1) * numcasos) for cual in range(numbloques)]
                limites[-1] = (limajaz-numcasos, limajaz);
                for bloq in limites:
                    ajuste.zero_grad()
                    sa = red(tea[bloq[0]:bloq[1], :])
                    ea = error(sa.reshape(tsa[bloq[0]:bloq[1]].shape), tsa[bloq[0]:bloq[1]])
                    if math.isnan(ea.item()):
                        print("Divergencia en iteración",it)
                        return None,None
                    else:
                        ea.backward()
                        ajuste.step()
                ajuste.zero_grad()
                salval = red(tev)
                ev = error(salval.reshape(tsv.shape), tsv)
                if evprevio < ev.item():
                        nvalfal = nvalfal + 1
                else:
                        nvalfal = 0
                        evprevio = ev.item()
                print('Iteración %r Error de validación %f\r'%(it,ev.item()), end="")
                if (args.paciencia>0):
                    ajustepaso.step(ev)
                if nvalfal > 5:
                    break
    except Exception as prob:
         print()
         print(prob)
         print ("Terminando en iteración", it)
    red.zero_grad()
    del tea,tsa,tev,tsv
    #Prueba
    salprured = red(tep)
    tfall = errorclas(salprured.reshape(tsp.shape), tsp)
    print ("Secuencia de optimización con ",menser(tfall), " en", it, "iteraciones")
    return float(tfall), red.cpu()
        
def unaprueba(argumentos):
    """
   lanza varios ajustes sobre el mismo conjunto de datos de entrenamiento
Recibe una permutación de los índices de datos, la cantidad de casos y la red
Devuelve la red de menor error y su error
    """
    global muesajuste,numuestras
    indices, numuestras, red = argumentos
    limajaz = int(muesajuste * numuestras)
    print ("Partición de ajuste con", limajaz, "casos")
    indices=array(indices)
    inda = indices[:limajaz]
    indv = indices[limajaz:]
    resuldat = []
    resulredes = []
    ninten = nintentos
    for inired in range(ninten):
        tfall, red = optiauxpar(red, limajaz, inda, indv)
        resuldat.append(tfall)
        resulredes.append(red)
    min_index, min_value = min(enumerate(resuldat), key=lambda p: p[1])
    print ("Partición de datos con ",menser(min_value))
    return resulredes[min_index], min_value

def multiprueba(red):
    """
   lanza varios intentos de red, variando el conjunto de datos
Recibe una red
Devuelve el conjunto de errores finales y redes ajustadas para cada partición de los datos
    """
    global numuestras,numpruebas
    indices = list(range(numuestras))
    pruebas = []
    for pru in range(numpruebas):
        random.shuffle(indices)
        pruebas = pruebas + [(copy.copy(indices), numuestras, red)]
    lisresul = map(unaprueba, pruebas)
    return list(lisresul)

def promedia(lisresul):
    """
Recibe el conjunto de errores finales y redes ajustadas para cada partición de los datos
Devuelve la mejor o FALSE si no ha podido sacar ninguna
    """     
    ert = 0
    n=0
    mejor=[0,1e9]
    for r in lisresul:
        try:
            if r[1]<mejor[1]:
                 mejor=r
            ert = ert + r[1]
            n+=1
        except Exception as e:
            print(e)
            print(r)
    if n>0:
        print ("Promedio de ",menser(ert / n))
        print ("Mejor error: ",mejor[1])
        if args.fichpesos:
            torch.save(mejor[0].state_dict(),args.fichpesos)
        #Y el profesor se quedó dudoso "Ya.. ¿pero cómo cargas esos pesos en otra red en otro programa?"
        # El AlumnoQueTodoLoSabía, imperturbable, "De esta forma
        #    red.load_state_dict(torch.load(nombredelfichero))
        #El profesor sintió curiosidad "¿Y podría guardar la red entera para cargarla sin tener que crearla previamente?"
        #La AlumnaQueTodoLoSabía asintió, "Sí, si están disponibles todas las definiciones de clase:
        #Guardar: torch.save(mejor[0],args.fichpesos[0])
        #Recuperar: red=torch.load(nombredelfichero)
        return mejor[0]
    else:
        print(lisresul)
        return False

if __name__ == '__main__':
    #######Parámetros que puedes tocar
    class params():
        def __init__(self):
            self.__dict__ = {
                'fichdatos': open('mnist_uint8.mat','rb'), #'fichero con los datos de imágenes')
                'minibatch': 10000, #'numero de casos usado en cada bloque de ajuste'
                'parts':1,#'número de pruebas de partición de la muestra')
                'opts':1,#'número de pruebas de inicialización/optimización con cada partición de muestra')
                'paciencia': 0, #'iteraciones de ajuste malas para cambiar tasa de aprendizaje' 0 es no cambiarla
                'dectasa': 0,#'si mayor que 0, factor de decrecimiento de tasa de aprendizaje si se agota la paciencia
                'momento': 0, #'momento para los algoritmos de ajuste que lo llevan 
                'limit': 100, #'límite de iteraciones de cada ajuste')
                'cpucuda':'cpu',#'usar núcleos de CPU', pero puedes poner cuda:0, cuda:1, ...
                'fichpesos':None, #'fichero para guardar pesos de red completa, por si quieres luego hacer pruebas con ello sin tener que repetir ajuste
                'learningrate': 0.005,
                'weightdecay':0,
            }
    args=params()
    #La prueba ya está separada. Partiremos sólo ajuste y validación
    a=7
    b=5
    c=3
    d=0
    muesajuste=0.9
    red=nn.Sequential(
        #lado0=28: 28x28x1
                    nn.LazyConv2d(a,3),
                    #L_out=>28-anc1+1xlado1xa
                    nn.LeakyReLU(),
                    nn.AvgPool2d(3), #capa reductora
                    nn.LazyConv2d(b,3), #Puedes poner LazyConv2(b,anc2)
                    #L_out=>lado1-anc2+1xlado2xb
                    nn.LeakyReLU(),
                    #L_out=>lado2/rxlado3xb 
                    nn.LazyConv2d(c,3),
                    #L_out=>lado3-anc3+1xlado4xc
                    nn.LeakyReLU(),
                    nn.Flatten(), #a esta altura deberia llegar 2x2 o 3x3
                   #lado4·lado4·c=num5

                    nn.LazyLinear(10),
                    nn.LeakyReLU(),

                    nn.LazyLinear(10),
                    nn.Sigmoid(),
        ).cpu()
        #Si has usado Lazy recuerda inicializar ahora, por ejemplo
    #definir error a optimizar
    error = nn.BCELoss()
    #error = nn.SmoothL1Loss()
    #error = nn.L1Loss()

    #definir algoritmo de ajuste
    #ajuste=torch.optim.LBFGS(red.parameters(),lr=...2,max_iter=10,history_size=80)
    #ajuste=torch.optim.Adagrad(red.parameters(),lr=...,weight_decay=0.005)
    algajuste=torch.optim.Adam#(red.parameters(),lr=0.01,weight_decay=0.00)#3)
    #ajuste=torch.optim.ASGD(red.parameters(),lr=...,weight_decay=0.004)
    #ajuste=torch.optim.RMSprop(red.parameters(),lr=...,weight_decay=0.001)
    #ajuste=torch.optim.Rprop(red.parameters(),lr=...)
    #ajuste = torch.optim.SGD(red.parameters(), lr=..., momentum=mom, weight_decay=..., nesterov=True)
  #########################################################
    if (args.paciencia > 0):
            ajustepaso = torch.optim.lr_scheduler.ReduceLROnPlateau(ajuste, patience=args.paciencia,
                                                                    factor=args.dectasa)
    #    ajustepaso=torch.optim.lr_scheduler.StepLR(ajuste,step_size=...,gamma=...)

    dtype = torch.float
    proc=args.cpucuda or 'cpu'#"cpu"#"cuda:0"
    print("Ejecutando en",proc)
    device=torch.device(proc)

    numpruebas =args.parts  # de partición de muestra
    nintentos = args.opts # de optimización
    
    #Cargamos los datos, todos
    matrices=sio.loadmat(args.fichdatos)
    datos=matrices['train_x'].astype(float)/255
    salidas=matrices['train_y'].astype(float)
    datpru=matrices['test_x'].astype(float)/255
    salpru=matrices['test_y'].astype(float)
    del matrices
    numuestras=len(datos)
    datos=datos.reshape((numuestras,1,28,28))
    datpru=datpru.reshape((10000,1,28,28))
    tep,tsp=torchea(datpru,salpru,dtype,device)
    #Los agrupamientos y búsqueda de vecinos próximos son muy problemáticos con muchos puntos y muchas dimensiones
    #Separaremos todo por azar

    red(tep)
    try:
        lisresul = multiprueba(red) 
    except Exception as problema:
        print (problema)
        probmens=problema.message.upper()
        if probmens.find('CUDA')>=0 and probmens.find('MEMORY')>=0:
            print ("Pasando a cpu")
            proc="cpu"
            device=torch.device(proc)
            lisresul = multiprueba(red)
    redfin=promedia(lisresul)
    if redfin:
         ejemplos(redfin)


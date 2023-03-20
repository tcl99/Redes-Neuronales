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
Recibe entradas y salidas, pasÃ¡ndolo a torch, al dispositivo indicado
  """
  redent=torch.tensor(entradas,dtype=tipo,device=donde)
  redsal=torch.tensor(salidas,dtype=tipo,device=donde)
  return redent,redsal

def errorclas(clasesal, clasereal): #Calcula el error en tanto por uno
    error = [clasesal[caso,:].argmax() != clasereal[caso,:].argmax() for caso in range(len(clasereal))]
    return sum(error) / float(len(clasereal))

def menser(error):
    """
Sacar la precisiÃ³n a partir del error (complementario)
>>>menser(0.1)
Aciertos=0.9
    """     
    return "Aciertos={}".format(1-error)

def mostrar():
              """
Mostrar un caso al azar: imagen de entrada, salida ideal y salida de la red
              """     
              global datos,salidas
              nmax=len(datos)-1
              i1=random.randint(0,nmax)
              salred = red(datos[i1:i1+1,:,:,:])
              imagorig=Image.fromarray(datos[i1:i1+1,:,:,:].detach().numpy().squeeze()*255)
              imagorig.resize((200,200),PIL.Image.BILINEAR).show()
              print("Real: {}".format(salidas[i1]),'Red: {}'.format(salred[0]))

def ejemplos(red):
   """
Control del ciclo de ir mostrando ejemplos con una red
   """
   seguir="s"
   while(seguir=="s"):
    mostrar()
    seguir=input("Teclea s si quierÃ©s mÃ¡s ")
	      
def optiauxpar(red0, limajaz, inda, indv,indp):
    """
   Pasada de optimizaciÃ³n
Recibe una red, la cantidad de casos de ajuste, sus Ã­ndices, los de validaciÃ³n y los de prueba
Devuelve el error final y la red ajustada
    """
    global datos,salidas,tep,tsp
    global device, error, ajuste, ajustepaso
    global it,evprevio, args
    if proc == "cpu":
        tea = datos[inda]
        tsa = salidas[inda]
        tev = datos[indv]
        tsv = salidas[indv]
        tep = datos[indp]
        tsp = salidas[indp]
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
                        print("Divergencia en iteraciÃ³n",it)
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
                print('IteraciÃ³n %r Error de validaciÃ³n %f\r'%(it,ev.item()), end="")
                if (args.paciencia>0):
                    ajustepaso.step(ev)
                if nvalfal > 5:
                    break
    except Exception as prob:
         print()
         print(prob)
         print ("Terminando en iteraciÃ³n", it)
    red.zero_grad()
    del tea,tsa,tev,tsv
    #Prueba
    salprured = red(tep)
    tfall = error(salprured.reshape(tsp.shape), tsp)
    print ("Secuencia de optimizaciÃ³n con ",menser(tfall.item()), " en", it, "iteraciones")
    return float(tfall), red.cpu()
        
def unaprueba(argumentos):
    """
   lanza varios ajustes sobre el mismo conjunto de datos de entrenamiento
Recibe una permutaciÃ³n de los Ã­ndices de datos, la cantidad de casos y la red
Devuelve la red de menor error y su error
    """
    global muesajuste,numuestras
    indices, numuestras, red = argumentos
    limajaz = int(muesajuste * numuestras)
    print ("ParticiÃ³n de ajuste con", limajaz, "casos")
    limvalaz = int(numuestras * (muesajuste + muesval))
    indices=array(indices)
    inda = indices[:limajaz]
    indv = indices[limajaz:limvalaz]
    indp = indices[limvalaz:]
    resuldat = []
    resulredes = []
    ninten = nintentos
    for inired in range(ninten):
        tfall, red = optiauxpar(red, limajaz, inda, indv,indp)
        resuldat.append(tfall)
        resulredes.append(red)
    min_index, min_value = min(enumerate(resuldat), key=lambda p: p[1])
    print ("ParticiÃ³n de datos con ",menser(min_value))
    return resulredes[min_index], min_value

class Media(nn.Module):
    """
MÃƒÂ³dulo consistente en varias cadenas de capas en paralelo, cuyas salidas deben tener las mismas dimensiones, porque se concatenan
    """
    def __init__(self,ramas):
        #ramas son los bloques separados que actÃƒÂºan
        #cada bloque es a su vez una secuencia
        super(Media, self).__init__()
        self.ramas = ramas

    def forward(self, x):
        #cada rama opera independientemente sobre el total y luego se concatenan sus resultados
        resuls=list()
        for bloque in self.ramas:
            resuls.append(bloque(x))
        return torch.cat(resuls,1)

def multiprueba(red):
    """
   lanza varios intentos de red, variando el conjunto de datos
Recibe una red
Devuelve el conjunto de errores finales y redes ajustadas para cada particiÃ³n de los datos
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
Recibe el conjunto de errores finales y redes ajustadas para cada particiÃ³n de los datos
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
        #Y el profesor se quedÃ³ dudoso "Ya.. Â¿pero cÃ³mo cargas esos pesos en otra red en otro programa?"
        # El AlumnoQueTodoLoSabÃ­a, imperturbable, "De esta forma
        #    red.load_state_dict(torch.load(nombredelfichero))
        #El profesor sintiÃ³ curiosidad "Â¿Y podrÃ­a guardar la red entera para cargarla sin tener que crearla previamente?"
        #La AlumnaQueTodoLoSabÃ­a asintiÃ³, "SÃ­, si estÃ¡n disponibles todas las definiciones de clase:
        #Guardar: torch.save(mejor[0],args.fichpesos[0])
        #Recuperar: red=torch.load(nombredelfichero)
        return mejor[0]
    else:
        print(lisresul)
        return False

if __name__ == '__main__':
    #######ParÃ¡metros que puedes tocar
    class params():
        def __init__(self):
            self.__dict__ = {
                'fichdatos': open('datos30.bin','rb'), #'fichero con los datos de imÃ¡genes')
                'minibatch': 1000, #'numero de casos usado en cada bloque de ajuste'
                'parts':1,#'nÃºmero de pruebas de particiÃ³n de la muestra')
                'opts':1,#'nÃºmero de pruebas de inicializaciÃ³n/optimizaciÃ³n con cada particiÃ³n de muestra')
                'paciencia': 0, #'iteraciones de ajuste malas para cambiar tasa de aprendizaje' 0 es no cambiarla
                'dectasa': 0,#'si mayor que 0, factor de decrecimiento de tasa de aprendizaje si se agota la paciencia
                'momento': 0, #'momento para los algoritmos de ajuste que lo llevan 
                'limit': 100, #'lÃ­mite de iteraciones de cada ajuste')
                'cpucuda':'cpu',#'usar nÃºcleos de CPU', pero puedes poner cuda:0, cuda:1, ...
                'fichpesos':None, #'fichero para guardar pesos de red completa, por si quieres luego hacer pruebas con ello sin tener que repetir ajuste
                'learningrate': 0.001,
                'weightdecay':0,
            }
    args=params()
    #La prueba ya estÃ¡ separada. Partiremos sÃ³lo ajuste y validaciÃ³n
    muesajuste=0.6
    muesval=0.25

    a=3
    b=2
    c=1
    anc=3
    red=nn.Sequential(
        nn.LazyConv2d(a,3),
        #L_out=>28-anc1+1xlado1xa
        nn.LeakyReLU(),
        #nn.AvgPool2d(3), #capa reductora

        nn.LazyConv2d(b,3), #Puedes poner LazyConv2(b,anc2)
        #L_out=>lado1-anc2+1xlado2xb
        nn.LeakyReLU(),
        #L_out=>lado2/rxlado3xb 
        nn.LazyConv2d(c,3),
        #L_out=>lado3-anc3+1xlado4xc
        nn.LeakyReLU(),
        
Media(nn.ModuleList([
            ##############Â¿Dos, tres, cuatro? ramas para intermedia. Rellenan para mantener el tamaÃƒÂ±o
            nn.Sequential(
                # ....
                # nn.Conv2d(nprevia,1,1),#reductora
                #  nn.LeakyReLU(),#no lineal
                # ....
                nn.LazyConv2d(3, anc, padding=anc//2, padding_mode= 'circular'),
                # convolutiva
                # ....
                nn.LeakyReLU(),  # no lineal
            ),
            nn.Sequential(
                # ....
                nn.LazyConv2d(3,anc),
                # ...x.
                nn.LeakyReLU(),#no lineal
            ),
            nn.Sequential(
                # ....
                #  nn.Conv2d(nprevia,1,1),#reductora
                #  nn.LeakyReLU(),#no lineal
                # ....
                nn.LazyConv2d( padding=anc//2, padding_mode=...),  
                # ....
                nn.LeakyReLU(),  # no lineal
            )]
        )),
        ################
        nn.LazyLinear(1)
    ).cpu()
    """
    red = nn.Sequential(
        # 240x320x1
        nn.AvgPool2d(...), # puede estar en otro sitio, pero fÃ­jate que las imÃ¡genes de entrada son grandes
        # L_out...
        nn.LazyConv2d(...), o Conv2d
        # L_out=>.....
        nn.LeakyReLU(),
        Media(nn.ModuleList([
            ##############Â¿Dos, tres, cuatro? ramas para intermedia. Rellenan para mantener el tamaÃƒÂ±o
            nn.Sequential(
                # ....
                # nn.Conv2d(nprevia,1,1),#reductora
                #  nn.LeakyReLU(),#no lineal
                # ....
                nn.LazyConv2d(numproc, ancho, padding=ancho//2, padding_mode= 'zeros', 'reflect', 'replicate' or 'circular'),
                # convolutiva
                # ....
                nn.LeakyReLU(),  # no lineal
            ),
            nn.Sequential(
                # ....
                nn.LazyConv2d(...),
                # ...x.
                nn.LeakyReLU(),#no lineal
            ),
            nn.Sequential(
                # ....
                #  nn.Conv2d(nprevia,1,1),#reductora
                #  nn.LeakyReLU(),#no lineal
                # ....
                nn.LazyConv2d( padding=anc//2, padding_mode=...),  
                # ....
                nn.LeakyReLU(),  # no lineal
            )]
        )),
        # ...
        nn.LazyConv2d(...),
        # L_out=>...
        nn.LeakyReLU(),
        nn.AvgPool2d(...), # O quizÃ¡ no
        # L_out...
        nn.Flatten(),
        # ...
        nn.LazyLinear(1), # La salida es 1
    ).cpu()
    """

    #definir error a optimizar
    error = nn.MSELoss() # Este problema es de regresiÃ³n no de clasificaciÃ³n
    #error = nn.SmoothL1Loss()
    #error = nn.L1Loss()

    #definir algoritmo de ajuste
    #ajuste=torch.optim.LBFGS
    #ajuste=torch.optim.Adagrad
    #algajuste=torch.optim.Adam#
    #ajuste=torch.optim.ASGD
    algajuste=torch.optim.RMSprop
    #ajuste=torch.optim.Rprop
    #ajuste = torch.optim.SGD
  #########################################################
    if (args.paciencia > 0):
            ajustepaso = torch.optim.lr_scheduler.ReduceLROnPlateau(ajuste, patience=args.paciencia,
                                                                    factor=args.dectasa)

    dtype = torch.float
    proc=args.cpucuda or 'cpu'#"cpu"#"cuda:0"
    print("Ejecutando en",proc)
    device=torch.device(proc)

    numpruebas =args.parts  # de particiÃ³n de muestra
    nintentos = args.opts # de optimizaciÃ³n
    
    #Cargamos los datos, todos
    datos, salidas = torch.load(args.fichdatos)
    numuestras=len(datos)
    #Los agrupamientos y bÃºsqueda de vecinos prÃ³ximos son muy problemÃ¡ticos con muchos puntos y muchas dimensiones
    #Separaremos todo por azar

    red(datos[0:1,:,:,:])
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

import torch
import torch.nn as nn
class Transforma1D(nn.Module):
    """
Módulo consistente en varias minirredes en paralelo, cada una trabajando con una entrada y concatenando sus salidas
    """
    def __init__(self,numentradas,numinter):
        #ramas son los bloques separados que actúan
        #cada bloque es a su vez una secuencia
        super().__init__()
        #Representa la red de trozos de red para cada variable del modelo lineal generalizado
        self.ramas = nn.ModuleList([nn.Sequential(nn.Linear(1, numinter),nn.Tanh(),nn.Linear(numinter, 1))]*numentradas)

    def forward(self, x):
        #cada rama opera independientemente sobre una entrada y luego se concatenan sus resultados
        resuls=list()
        for ent in range(len(self.ramas)):
            resuls.append(self.ramas[ent](x[:,ent:ent+1]))
        return torch.cat(resuls,1)


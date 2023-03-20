# Redes profundas

Vamos a medir: - Tiempo - Error - Estabilidad

LBFGS y SGD no llevan control de derivadas y los demás sí

Si la red no es grande, estos algoritmos son mejores, sólo cuando las derivadas se disparan y las redes son grandes realmente ayudan los otros que tienen control de derivadas

## NORMALIZAR

Para normalizar entre capa y capa metemos el tochaco

Cuanto mas bajo es el minibatch tarda más, pero es más estable y controla mejor el error
La idea general es probar con uno grande e ir bajando

## HIPóTESIS(Son creibles en cierta medida)

### Los 3 primeros van sobre los datos

1. resultados=f(entradas) + ruido impredecible
   El ruido:
   - Su media va a 0
   - Su varianza es constante
   - Sin correlación ni dependencia(ni de entradas ni consigo), es independiente
   - probabilidad
1. Tenemos una idea de esa probalididad
1. La muestra que tenemos representa totalmente la realidad

### A partir del 4 REDES

4. Si hacemos cualquier cambio de pesos en la red la respuesta cambiará, normalmente
5. Existe un conjunto de pesos para el que clava perfectamente la red (f(entradas)), el objetivo es encontrarlo
6. Conocemos la distribción normal de probabilidad de los pesos buenos

Para cada conjunto de pesos, hay una probablidad de que clave la muestra, se busca subir desde los pesos que tengamos a los ideales,
Deberiamos minimizar el error cuadrático y optimizar los pesos pa q la red sea buena
el weight_decay a veces va bien, ha de ser pequeño normalmente (0.001 o así) y en una red grande

## Desactivar procesadores

Redes deep, ejemplo con la de los numeros

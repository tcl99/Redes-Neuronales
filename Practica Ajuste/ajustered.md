# PRACTICA 3 AJUSTE DE VALORES

## Cómo decidir que valores ponerle a la red

1. PARÁMETROS DE PARTICION DE MUESTRA
   AJUSTE AZAR VAL
2. NUMERO DE PROCESADORES OCULTOS
   VAROCULTOS
3. FUNCION DE ERROR A MINIMIZAR
   MSELOSS L1LOSS TAL

4. NORMALIZAR
5. ALGORITMO DE AJUSTE
   LR
   CUASI-NEWTON
   GRADIENTE ESTOCASTICO
   PASO FIJO
   APROXIMACION MODULO 1

   SEGÚN LA MUESTRA, EL RENDIMIENTO DE LOS ALGORITMOS VARÍA

   - Cuasi newton lleva mal las muestras grandes, mejor para medianos
   - Paso fijo y aproximacion modulo son variaciones del gradiente estocastico para intentar mejorar, para problemas grandes suelen ser mejor estos 2
   - Si se baja el learning rate, baja el tiempo pero se pierde estabilidad

## ESTIMAR ERROR FUTURO = GENERALIZACION

### ESTRATEGIAS:

1.  UN CONJUNTO DE PRUEBA (Sobre datos originales) (La primera prueba)
2.  CONJUNTO DE DATOS SINTETICO (La prueba del fichero aumentado)
    Abundancia de todos los casos - Sí, igual o peor - No, mejor si aciertas con la generalización buena
3.  ¿Mínimo? ¿Promedio? Mucho mejor quedarse con el promedio (Lanzar varias pruebas para quedarse con el promedio)
4.  Promedio de muestreo con reemplazo (BOOTSTRAP)
    La idea del bootstrap es, tenemos una muestra total con una parte con reemplazo: - Sin reemplazo(Sacar pelota de la bolsa y meterla) - Con reemplazo(Sacar pelota de la bolsa y no meterla) Usaremos esta
    Del reemplazo de saca el ajuste y del ajuste se saca la red, con la muestra total, le aplicas la red y observas el error

    #EN ESTA PARTE ES DONDE SACAS LA MUESTRA, EL TAMAÑO DE LA MUESTRA TOTAL SE ELIGE
    muestras=data.DataLoader(total,batch_size=tamsubmues, sampler=selec) bbotstrap? con reemplazo y sin reemplazo, generalmente con
    muestras=data.DataLoader(total,batch_size=numuestras//numcrosval, shuffle=True) validación cruzada

    Tras las pruebas realizadas con modelos.py, se puede deducir que en general funciona y algún valor que se va, por lo que tenemos que apreciar la barra más grande

    Usando la validación cruzada(ejemplo de la pizza), se coge una porción, se hace el ajuste con el resto de pizza y la prueba con la porción cogida. Cada vez se hace con una porción distinta, haciendo la prueba con cada uno de los trozos una vez, y con el resto el ajuste, si se parte en 8 trozos, se realiza una vvez con cada trozo

MÉTODO 3

Hasta ahora estabamos separando la red, y no la dejabamos aprender con toda la red, hay un problema, la red nunca tiene toda la info. Se hace para poder estimar la generalización.

¿Hay formas de darle odos los datos y poder estimar la genralizacion?

Se puede y se divide en 2 fases

- Fase 1:
  Se coge una red grande, separando varias veces el conjunto de prueba
  Se realizan varias particiones ajuste prueva val
  En media donde para el ajuste
  La prueba dondeva
- Fase 2:
  Ahora cojo una red, la que sea, red mia, y le doy todo el conjunto de ajuste
  Cuando hace el ajuste le paro donde la primera vez
  Estimo generalización sumando las pruebas anteriores de la red grande

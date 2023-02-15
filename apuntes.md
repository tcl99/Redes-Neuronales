# Apuntes Introducción

En las redes pequeñas las capas adyacentes suelen estar completamente conectadas, en las grandes no
Lo bueno de la tanh es que está acotada entre 1 y -1
es monótona, cada x tiene su y
tiene una zona, la del medio, que es prácicamente lineal

Las redes operan en práctica matricialmente calculando todos los pesos para una gran cantidad de entradas, por esto es interesante observar implementaciones como CUDA que utilizan la GPU(optimizada para este proceso), que agilizan el proceso de cálculo

Método para entrenar una red con una muestra:

MUESTRA se suele dividir en: AJUSTE, VALIDACIÓN Y PRUEBA
(NO ES OBLIGATORIO PERO ES POPULAR)

AJUSTE: Aprendizaje de la red
VALIDACIÓN: Para verificar el AJUSTE (Hay otras opciones)
PRUEBA: Para estimar la generalización (Si se puede extrapolar la red a otras muestras) (Hay también otras opciones)

Hay que tener en cuenta la distribución de los datos, puede no ser uniforme, cuando no lo es, tomar muestras aletoatoriamente puede dejar de lado casos extremos
Muestreo estratificado, hacerlo al azar pero por grupos

Todos los algoritmos de redes son iterativos.
Nivel de acierto mínimo, hay aplicaciones de las redes donde un 3% de error es más que aceptable y otras donde el mismo 3% es una mierda

Se puede calcular el error minimo adecuado aplicando modelos lineales por ejemplo

Saber cuándo aplicar una red neuronal, a veces aplicarla es como matar moscas a cañonazos, por eso probar el modelo lineal. NORMALMENTE la red neuronal va mejor que el modelo lineal lógicamente

Con el modelo linel obtengo resultados de error de 0,5
Con la red neuronal entre 0,3 y 0,4

En el modelo lineal generalizado, las variables son transformadas, estas funciones se pueden aplicar con redes neuronales a cada variable
MODELO 2

REPETIR VARIAS VECES CADA PRUEBA

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

1.  UN CONJUNTO DE PRUEBA (Sobre datos originales)
2.  CONJUNTO DE DATOS SINTETICO
    Abundancia de todos los casos - Sí, igual o peor - No, mejor si aciertas con la generalización buena
3.  ¿Mínimo? ¿Promedio? Mucho mejor quedarse con el promedio
4.  Promedio de muestreo con reemplazo (BOOTSTRAP)
    La idea del bootstrap es, tenemos una muestra total con una parte con reemplazo: - Sin reemplazo(Sacar pelota de la bolsa y meterla) - Con reemplazo(Sacar pelota de la bolsa y no meterla) Usaremos esta
    Del reemplazo de saca el ajuste y del ajuste se saca la red, con la muestra total, le aplicas la red y observas el error

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 8 16:36:12 2026

@author: physicist

Implementación del método del rechazo de una distribución uniforme 
a una no uniforme.
"""

# Llamada a funciones externas
from math import exp
from pylab import show, plot, hist, figure, xlabel, ylabel, title
from numpy import zeros
from random import seed, random

# Declaración o inicialización de variables
N = 100000                         # Cantidad de números a generar
y = zeros(N, float)                # Colección para la distribución uniforme
P = zeros(N, float)                # Colección de elementos de comparación 
yg = []                            # Colección para la distribución no uniforme

# P(y) = B * exp(-(y - yc)^2/sigma^2) # Distribucion de Gauss a obtener

B, yc, sigma = 1, 5, 2 # Inicialización de variables de la distribución deseada

seed(10)                           # Semilla

for i in range(N):      # Bucle que genera los aleatorios y compara cada uno
    y[i] = 10*random()  # Distribución original
    P[i] = B * exp(-(y[i] - yc)**2 / sigma**2) # Distribución para comparar
    prueba = random()   # Aleatorio en ordenadas para comparar
    if(P[i] > prueba):  # Comparación y aceptación de números cuyo valor
        yg.append(y[i]) # en P(x) están por debajo de la curva

figure(3)               # Representación de la distribución no uniforme
hist(y, bins = 100)
hist(yg, bins = 100)
xlabel("Número generado")
ylabel(r"$P(y) = B * exp(-(y - yc)^2/sigma^2)$")
title('Distribución No Uniforme')
show()

figure(2)               # Representación de la función a obtener
plot(y, P, "o")
xlabel("Número generado")
ylabel(r"$f(y) = B * exp(-(y - yc)^2/sigma^2)$")
title('Función de la Distribución a Obtener')
show()

figure(1)               # Representación de la función uniforme
hist(y, bins = 100)
xlabel("Número generado")
ylabel("Número de veces")
title('Distribución Uniforme')
show()

# Comparación entre el número de elementos en ambas distribuciones.
print()
print(len(y))
print()
print(len(yg))

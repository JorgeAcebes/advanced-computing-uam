# -*- coding: utf-8 -*-
"""
Created on Sun Mar 8 16:22:10 2026

@author: physicist

Implementación del método de transformación de una distribución uniforme 
a una no uniforme.
"""

# Llamada a funciones externas
from math import log
from pylab import show, hist, figure, xlabel, ylabel
from numpy import zeros
from random import random

# Declaración o inicialización de variables
N = 10000                   # Cantidad de números a generar
x = zeros(N, float)         # Colección para la distribución uniforme
y = zeros(N, float)         # Colección para la distribución no uniforme

# P(y) = exp(-y)            # Distribución de Poisson que queremos obtener

for i in range(N):          
    x[i] = random()         # Inicialización de la distribución uniforme
    y[i] = - log(x[i])      # Transofrmación a distribución no uniforme


figure(2)                   # Representación de la distribución transformada
hist(y, bins = 100)
hist(x, bins = 100)
xlabel("Número generado")
ylabel(r"$P(y) = exp(-y)$")
show()

figure(1)                   # Representación de la distribución original
hist(x, bins = 100)
xlabel("Número generado")
ylabel("Número de veces")
show()
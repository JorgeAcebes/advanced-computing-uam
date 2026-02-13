import numpy as np
import matplotlib.pyplot as plt


def Catalan_nums():
    C = 1.0
    n = 0
    print('='*50)
    print('Números de Catalan hasta "1 billion" (mil millones)')
    while C <= 1e9:
        print(C)
        C = (4*n+2)*C/(n+2)
        n += 1
    print('='*50)
Catalan_nums()

# EXTRA: 

def Catalan_constant():
    G  = 0
    for n in np.arange(1e6):
        G += (-1)**n / (2*n+1)**2

    return G

G = Catalan_constant()

print('='*50)
print('Constante de Catalan hasta la duodécima posición:')
print(f"{G:.12f}")
print('='*50)
 
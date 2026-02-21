# %%
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
# %%

def distancia(x,y,z):
    return(np.sqrt(x**2+y**2+z**2))

# %%
def potencial(x,y,z,a):
    '''
    Calcula potencial eléctrico en el origen creado por un ión

    a: distancia al origen
    '''
    e = sp.constants.elementary_charge
    pi = sp.constants.pi
    e_0 =sp.constants.epsilon_0
    charge = +1 if (x+y+z) % 2 == 0 else -1
    if a == 0:
        print('No se puede calcular el potencial en el origen')
        return None
    else:
        V = charge * e / (4 *pi * e_0 * a *distancia(x,y,z))


# %%
def Madelung(L):
    '''
    Calcula la constante de Madelung para un cubo de lado 2L centrado en el origen.
    
    L: semilongitud del cubo
    '''
    M= 0
    x = np.arange(-L,L+1)
    for i in x:
        for j in x:
            for k in x:
                if (i,j,k) == (0,0,0): continue                
                signo = 1 if (i+j+k)%2 == 0 else -1 
                M += signo / distancia(i,j,k)
    return M

print(Madelung(100))

# %%
def madelung_better(L):
    r = np.arange(-L,L+1)
    x,y,z = np.meshgrid(r,r,r,indexing='ij')
    dist = np.sqrt(x**2 + y **2 + z**2)
    mask = dist != 0
    sign = np.where((x+y+z)%2 ==0, 1.0, -1.0)

    M = np.sum(sign[mask] / dist[mask])
    return M

madelung_better(500)


# %%

L = 1
r = np.arange(-L,L+1)
x,y,z = np.meshgrid(r,r,r,indexing='ij')
dist = np.sqrt(x**2 + y **2 + z**2)
mask = dist != 0
sign = np.where((x+y+z)%2 ==0, 1.0, -1.0)
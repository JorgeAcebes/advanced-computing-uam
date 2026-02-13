import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

 
m_p = sp.constants.m_p
h = sp.constants.hbar
eV = sp.constants.electron_volt

def T_R(m, E, V):
    '''
    Calcular coeficientes de transmisión y reflexión para potencial escalón cuántico

    m: masa (kg)
    E: energía (eV)
    V: potencial tras escalón (eV)
    '''
    if E < V: 
        return 0, 1
    k1 = np.sqrt(2*m*E*eV) / h
    k2 = np.sqrt(2*m*(E-V)*eV) / h

    T = 4*k1*k2/(k1+k2)**2
    R = ((k1-k2)/(k1+k2))**2
    return T, R


E = 10
V= 9

T, R = T_R(m_p,E,V)

print('='*40)
print(f"Coeficiente de Transmisión (T): {T:.3f}")
print(f"Coeficiente de Reflexión   (R): {R:.3f}")
print('='*40)


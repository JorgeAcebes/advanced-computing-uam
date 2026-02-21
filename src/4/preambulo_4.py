# %% Imports y Declaraciones

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy as sp
import sys
from pathlib import Path
import importlib

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))

import tools as ja

importlib.reload(ja)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.serif": ["DejaVu Serif"],
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15
})



# Función para dar formato a los plots
def espacio_tiempo(ax):
    ax.set_xlabel(r'Tiempo $t$ [s]')
    ax.set_ylabel(r'Ángulo $\theta$ [rad]')
    # Detalles finos (Grilla y Ticks)
    ax.grid(True, which='major', linestyle='-', alpha=0.6, color='#d3d3d3')
    # Leyenda (para eso sirve label)
    ax.legend(frameon=True, loc='best', fontsize=12, framealpha=0.6, edgecolor='gray')


def espacio_fases(ax):
    ax.set_xlabel(r'Posición $x$ [m]')
    ax.set_ylabel(r'Velocidad $v$ [m/s]')
    # Detalles finos (Grilla y Ticks)
    ax.grid(True, which='major', linestyle='-', alpha=0.6, color='#d3d3d3')
    # Leyenda (para eso sirve label)
    ax.legend(frameon=True, loc='best', fontsize=12, framealpha=0.6, edgecolor='gray')



# Definición de funciones y constantes:

g = sp.constants.g
l = 1
q = 0.5
F_D = 2


def restringir(o):
    o = (o + np.pi) % (2*np.pi) - np.pi # Restrinjo el ángulo entre -pi y pi
    return o

def pend_lin_forz_fric(t, state, Omeg):
    o = state[0]


    w = state[1]

    a = -g/l * o - q * w + F_D * np.sin(Omeg*t)

    return np.array([w,a])

def pend_NL_forz_fric(t,state, Omeg):
    o = state[0]
    w = state[1]

    a = - g/l * np.sin(o) - q * w + F_D * np.sin(Omeg * t)

    return np.array([w,a])

# %% === Ejecución y Gráficos


y0 = [np.pi/100, 0]

t = np.linspace(0,100,2*10**3)

O = np.arange(0.1,6,0.05)
A = []

A_NL =[]

for i, Omeg in enumerate(O):

    pend_LFF = lambda t, y: pend_lin_forz_fric(t,y,Omeg)
    pend_NLFF = lambda t,y: pend_NL_forz_fric(t,y, Omeg)

    sol_e_c = ja.rk4_solver(pend_LFF, t, y0)
    sol_e_c_NL = ja.rk4_solver(pend_NLFF, t, y0)

    o, _ = sol_e_c.T

    o = restringir(o)

    o_NL, _ = sol_e_c_NL.T

    o_NL = restringir(o_NL)

    long_estacionaria = len(o)//2

    amp = (np.max(o[long_estacionaria:]) - np.min(o[long_estacionaria:]))/2
    amp_NL = (np.max(o_NL[long_estacionaria:]) - np.min(o_NL[long_estacionaria:]))/2

    A.append(amp)
    A_NL.append(amp_NL)


fig, ax = plt.subplots(1, 2, figsize=(16, 5), dpi=120)

ax[0].plot(O,A, 'o-', color = 'violet',  markersize = 4)

ax[0].axvline(np.sqrt(g/l), color = 'red', linestyle = ':', label = r'$\sqrt{g/l}$')
ax[0].axvline(np.sqrt(g/l - q**2/2 ), color = 'green', linestyle = ':', label = r'$\sqrt{g/l-q^2/2}$')
ax[0].legend()


ax[0].set_xlabel(rf'$\Omega_D$ [rad]')
ax[0].set_ylabel(rf'Amplitud')
ax[0].set_title(rf'Amplitud en función de $ \Omega_D$ $-$ Caso Lineal')



ax[1].plot(O,A_NL, 'o-', color = 'violet',  markersize = 4)

ax[1].axvline(np.sqrt(g/l), color = 'red', linestyle = ':', label = r'$\sqrt{g/l}$')
ax[1].axvline(np.sqrt(g/l - q**2/2 ), color = 'green', linestyle = ':', label = r'$\sqrt{g/l-q^2/2}$')
ax[1].legend()


ax[1].set_xlabel(rf'$\Omega_D$ [rad]')
ax[1].set_ylabel(rf'Amplitud')
ax[1].set_title(rf'Amplitud en función de $ \Omega_D$ $-$ Caso No Lineal')




plt.tight_layout()
plt.show()
plt.close()

# %%


A_max = np.max(A)
O_max = O[np.argmax(A)]


pend_LFF = lambda t, y: pend_lin_forz_fric(t,y,O_max)
sol_e_c_max = ja.rk4_solver(pend_LFF, t, y0)

o_max, w_max = sol_e_c_max.T


O_other = float(O[np.isclose(O, 2, atol=1e-2)])

pend_LFF = lambda t, y: pend_lin_forz_fric(t,y,O_other)
sol_e_c_other = ja.rk4_solver(pend_LFF, t, y0)

o_other, w_other = sol_e_c_other.T


fig, ax = plt.subplots(2,2, figsize=(16, 12), dpi=120)

ax[0,0].plot(t,o_max,color ='blueviolet', linestyle = '--', label = rf'$\Omega_D = {O_max:.2f}$ rad (resonante)', alpha = 0.5)
ax[0,0].plot(t,o_other,color ='orange', linestyle = '--', label = rf'$\Omega_D = {O_other:.2f}$ rad', alpha = 0.5)


ax[0,1].plot(o_max, w_max, color ='blueviolet', linestyle = '--', label = rf'$\Omega_D = {O_max:.2f}$ (resonante)', alpha = 0.5)
ax[0,1].plot(o_other, w_other, color ='orange', linestyle = '--', label = rf'$\Omega_D = {O_other:.2f}$', alpha = 0.5)


espacio_tiempo(ax[0,0])
espacio_fases(ax[0,1])


ax[0,0].legend()
ax[0,0].set_title('Ángulo en función del tiempo $-$ Caso Lineal')

ax[0,1].legend()
ax[0,1].set_title('Espacio de fases $-$ Caso Lineal')



A_max_NL = np.max(A_NL)
O_max_NL = O[np.argmax(A_NL)]


pend_NLFF = lambda t, y: pend_lin_forz_fric(t,y,O_max_NL)
sol_e_c_max_NL = ja.rk4_solver(pend_NLFF, t, y0)

o_max_NL, w_max_NL = sol_e_c_max_NL.T


O_other_NL = float(O[np.isclose(O, 2, atol=1e-2)])

pend_NLFF = lambda t, y: pend_lin_forz_fric(t,y,O_other_NL)
sol_e_c_other_NL = ja.rk4_solver(pend_NLFF, t, y0)

o_other_NL, w_other_NL = sol_e_c_other_NL.T


ax[1,0].plot(t,o_max_NL,color ='blueviolet', linestyle = '--', label = rf'$\Omega_D = {O_max_NL:.2f}$ rad (resonante)', alpha = 0.5)
ax[1,0].plot(t,o_other_NL,color ='orange', linestyle = '--', label = rf'$\Omega_D = {O_other_NL:.2f}$ rad', alpha = 0.5)


ax[1,1].plot(o_max_NL, w_max_NL, color ='blueviolet', linestyle = '--', label = rf'$\Omega_D = {O_max_NL:.2f}$ (resonante)', alpha = 0.5)
ax[1,1].plot(o_other_NL, w_other_NL, color ='orange', linestyle = '--', label = rf'$\Omega_D = {O_other_NL:.2f}$', alpha = 0.5)


espacio_tiempo(ax[1,0])
espacio_fases(ax[1,1])


ax[0,0].legend()
ax[0,0].set_title('Ángulo en función del tiempo')

ax[0,1].legend()
ax[0,1].set_title('Espacio de fases')


ax[1,0].legend()
ax[1,0].set_title('Ángulo en función del tiempo $-$ Caso No Lineal')

ax[1,1].legend()
ax[1,1].set_title('Espacio de fases $-$ Caso No Lineal')


plt.tight_layout
plt.show()
plt.close()



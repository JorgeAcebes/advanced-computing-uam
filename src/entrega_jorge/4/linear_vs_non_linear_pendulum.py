'''
This is not useful for the report.
'''
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
ja.setup_style()


# Definición de funciones y constantes:

# --- Funciones para integrar ---
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

# --- Constantes y Variables ---
g = sp.constants.g
l = 1 #[m] 
q = 0.25
F_D = 2
t_tot = 30
dt = 0.01
ang0 = 5

# Valores inciales 
y0 = [np.radians(ang0), 0]
t = np.arange(0,t_tot,dt)
O = np.arange(0.1,6,0.1)

# === Ejecución y Gráficos

# Incializamos vectores para buscar amplitud en caso lineal y no lineal
A, A_NL = [], []

# Calculamos ángulos y amplitudes
for Omeg in O:
    for A_list, func in zip((A, A_NL), (pend_lin_forz_fric, pend_NL_forz_fric)):
        sol = ja.rk4_solver(lambda t, y: func(t,y,Omeg), t, y0)
        o = ja.restringir(sol.T[0])
        A_list.append(ja.amplitud(o, (t_tot-5)/t_tot)) # Hardcodeamos un valor de 5 segundos para la fase transitoria

A, A_NL = np.array(A), np.array(A_NL)

# Gráficas
fig, axes  = plt.subplots(1, 2, figsize=(16, 5), dpi=120)
title_type = ['Lineal', 'No Lineal']
w_0, O_D = np.sqrt(g/l), np.sqrt(g/l - q**2/2)

for ax, Amp, title_typ in zip (axes, [A, A_NL], title_type):
    ax.plot(O,Amp, 'o-', color = 'violet',  markersize = 4)
    ax.axvline(w_0, color = 'red', linestyle = ':', label = r'$\omega_0 = \sqrt{g/l}$')
    ax.axvline(O_D, color = 'red', linestyle = ':', label = r'$\Omega_D = \sqrt{g/l - q^2/2}$')
    ja.setup_ax(ax, title=rf'Amplitud en función de $\Omega_D$ $-$ Caso {title_typ}', xlabel=r'$\Omega_D$ [rad]', ylabel='Amplitud')

plt.tight_layout()
plt.show()
plt.close()

# %% Espacio de Fases

fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=120)

O_oth = O[np.argmin(np.abs(O-2))]

for i, (A_list, func, title_typ) in enumerate(zip((A, A_NL), (pend_lin_forz_fric, pend_NL_forz_fric), title_type)):
    O_max = O[np.argmax(A_list)]

# Resolver ambos casos de frecuencia
    sol_max = ja.rk4_solver(lambda t, y: func(t, y, O_max), t, y0)
    sol_other = ja.rk4_solver(lambda t, y: func(t, y, O_oth), t, y0)
    
    o_max, w_max = sol_max.T
    o_oth, w_oth = sol_other.T
    
    # Plot Espacio-Tiempo (Columna 0)
    ja.multiline_plot(axes[i, 0], t, [o_max, o_oth], 
                   labels=[rf'$\Omega_D = {O_max:.2f}$ rad (resonante)', rf'$\Omega_D = {O_oth:.2f}$ rad'],
                   colors=['blueviolet', 'orange'], styles=['--', '--'], alpha=0.5)
    ja.setup_ax(axes[i, 0], title=rf'Ángulo en función del tiempo $-$ Caso {title_typ}', 
             xlabel=r'Tiempo $t$ [s]', ylabel=r'Ángulo $\theta$ [rad]')
    
    # Poincare y Plot Espacio de Fases (Columna 1)
    o_max, w_max, _ = ja.sec_poincare_forz(o_max, w_max, O_max, t)
    o_oth, w_oth, _ = ja.sec_poincare_forz(o_oth, w_oth, O_oth, t)
    
    ja.multiscatter_plot(axes[i, 1], [o_max, o_oth], [w_max, w_oth],
                      labels=[rf'$\Omega_D = {O_max:.2f}$', rf'$\Omega_D = {O_oth:.2f}$'],
                      colors=['blueviolet', 'orange'], alpha=0.5)
    ja.setup_ax(axes[i, 1], title=rf'Sección de Poincaré $-$ Caso {title_typ}', 
             xlabel=r'Posición angular $\theta$ [rad]', ylabel=r'Velocidad angular $\omega$ [rad/s]')

plt.tight_layout()
plt.show()



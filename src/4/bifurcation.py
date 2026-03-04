# %% Imports y Declaraciones
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy as sp
import sys
from pathlib import Path
import importlib

figuras = Path(__file__).resolve().parent.parent / '4' / 'figures'
figuras.mkdir(parents=True, exist_ok=True)  

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
import tools as ja
importlib.reload(ja)
ja.setup_style()

# Definición de funciones y constanes:

# --- Funciones para integrar ---

def pend_NL_forz_fric(t,state, F_D):
    o = state[0]
    w = state[1]
    a = - np.sin(o) * g/l  - q * w + F_D * np.sin(Omeg * t)
    return np.array([w,a])

# --- Funciones auxiliares --- 

def period_number(vector_state, tol = 1e-4):
    unique_states = []
    for value in vector_state:
        if not any(np.abs(value - unique) < tol for unique in unique_states):
            unique_states.append(value)
    T = len(unique_states) # La cantidad de estados únicos nos dice la cantidad de periodos
    return T

def detect_change_print(Force_List, T, ax, F_D_max, color = 'black'):
    '''
    Detecta cuándo ha ocurrido un cambio de periodo, y pinta una línea vertical donde haya ocurrido
    '''
    T_curr = 0
    updated = 0
    F_change = []

    for i, _ in enumerate(Force_List): 
        if Force_List[i] > F_D_max:
            break
        else:
            if np.abs(T_curr - T[i]) != 0:
                T_curr = T[i]
                if i not in [0,1]: # Quito el primer elemento porque empiezo con un periodo falso de 0, y quito el segundo elemento porque ese lo empleamos para detectar el atractor, puede contener datos espurios
                    if updated == 0: # Le exijo cierta continuidad, no quiero que detecte "nuevas bifuraciones" continuamente, será posiblemente que no se ha estabilizado
                        ax.axvline(Force_List[i-1], color = color , ls = '-.' ) # Hago que la línea esté en el paso previo, que es el último punto que cambia
                        F_change.append(Force_List[i-1])
                        updated = 1
            else:
                updated = np.max([0, updated-0.5]) 

    return F_change

def calculate_feigenbaum(F_changes):
    long = len(F_changes)

    if long < 3:
        print('No se tienen suficientes F_n para calcular el número de Feigenbaum. Se requieren al menos tres valores.')
        return None

    feigenbaums = []  
    for i in np.arange(1, long-1):
        feig = (F_changes[i] - F_changes[i-1]) / (F_changes[i+1] - F_changes[i])
        feigenbaums.append(feig)

    return feigenbaums

# --- Constantes y Variables ---
g = sp.constants.g
l = g #[m] 
q = 0.5

dDriving = 0.001
Omeg = 2/3

N_periods = 200
t_tot = 2 * np.pi * N_periods / Omeg # Hacemos que el tiempo total sea un múltiplo del periodo.
dt = 0.01

t = np.arange(0,t_tot,dt)


ang = np.degrees(0.2)
y0 = [np.radians(ang), 0]

F_D_list = np.arange(1.35, 1.51, dDriving)


calcular_again = 0 # Determina si realizar el cálculo numerico de nuevo o si obtengo los resultados directamente del txt


# Bifuraction

if calcular_again:

    o_last = []
    w_last = []
    y_curr = y0
    F_plot = []
    T_o =    []
    T_w =    []


    for i, F_D in enumerate(F_D_list):
        sol = ja.rk4_solver(lambda t, y: pend_NL_forz_fric(t, y, F_D), t, y_curr)
        o, w = sol.T
        o = ja.restringir(o)
        o, w, _ = ja.sec_poincare_forz(o, w, Omeg, t)


        periodo_o = period_number(o)

        T_o.append(periodo_o)

        periodo_w = period_number(w)

        T_w.append(periodo_w)

        y_curr = [o[-1], w[-1]]

        n_last = 30 # Nos quedaremos con los útlimos n_last resultados
        o_last.extend(o[-n_last:]) 
        w_last.extend(w[-n_last:])
        F_plot.extend([F_D] * len (o[-n_last:])) # Para no tener problemas de dimensiones, generamos un F_D más grande

    data_matrix = np.column_stack((o_last, w_last, F_plot))
    time_matrix = np.column_stack((T_o, T_w))

    np.savetxt('bifurcation.txt', 
             data_matrix, 
             header='o \t w \t F_D',
            comments='')
    
    np.savetxt('bifurcation_time.txt', 
             time_matrix, 
             header='T_o \t T_w',
            comments='')


#  Desempaquetamos los resultados
o_last, w_last, F_plot = np.loadtxt('bifurcation.txt', skiprows=1, unpack=True)
T_o, T_w = np.loadtxt('bifurcation_time.txt', skiprows=1, unpack=True)


fig, ax = plt.subplots(1,2, figsize=(24,8), dpi = 120)

ax[0].scatter(F_plot, o_last, color = 'violet', alpha = 0.5)
ax[1].scatter(F_plot, w_last, color = 'lawngreen', alpha = 0.5)


ja.setup_ax(ax[0], r'Diagrama de bifurcación para $\theta$', xlabel = r'Driving Force $F_D$', ylabel = r'Ángulo $\theta$')
ja.setup_ax(ax[1], r'Diagrama de bifurcación para $\omega$', xlabel = r'Driving Force $F_D$', ylabel = r'Velocidad Angular $\omega$')

for i in [0,1]:
    ax[i].set_xlim(1.35, 1.495)

ax[0].set_ylim(0.9, 3.2)
ax[1].set_ylim(-2.2, -1.1)



# Intento insatisfactorio de calcular el número de Feigenbaum. Se requeriría mayor resolución

feigenbaums =  [[], []]

for i, T in enumerate([T_o, T_w]):
    F_changes = detect_change_print(F_D_list, T, ax[i], F_D_max = 1.48) #Limitamos para que pare a cuando empieza el régimen caótico (\lambda > 0).
    feigenbaums[i] = calculate_feigenbaum(F_changes)


plt.savefig(figuras/f'bifurcation.png', bbox_inches='tight') # Formato vectorial preferido


plt.tight_layout()
plt.show()


feigenbaums


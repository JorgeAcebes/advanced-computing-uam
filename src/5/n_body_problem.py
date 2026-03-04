# %% Imports y Declaraciones
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy as sp
import sys
from pathlib import Path
import importlib
from scipy.stats import linregress
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
import tools as ja
importlib.reload(ja)
ja.setup_style(base_size=19, dpi = 120)


# === Decidir qué se grafica ===
no_interaction = 0
w_interaction  = 1

# --- Variables y Funciones ---

GM_S = 4 * np.pi ** 2
GM_T = GM_S / 332946
GM_J = GM_S / 1048

r_S = 1.0 # Distancia Sol-Sol
r_T = 1.0 # Distancia Tierra-Sol 
r_J = 5.2 # Disntancia Jupiter-Sol

# Periodos
T_T = 1.0
T_J = 11.86

v_S = 0
v_T = 2 * np.pi * r_T / T_T
v_J = 2 * np.pi * r_J / T_J


# Vector de Tiempo
t_tot = 200
dt = 0.01
t = np.arange(0, t_tot, dt)


# -- Funciones --

def grav_2_bodies(t,state):
    x, y, = state[0], state[1]
    vx, vy = state[2], state[3]

    r3 = np.linalg.norm([x, y])**3

    ax = - GM_S * x / r3
    ay = - GM_S * y / r3

    return np.array([vx, vy, ax, ay])


def grav_n_bodies(t, state, masas):
    N = len(masas)
    
    # state = [x1, y1, z1, ... xN, yN, zN, vx1, vy1, vz1, ..., vxN, vyN, vzN]
    r = state[:3*N].reshape((N, 3))
    v = state[3*N:]
    
    a = np.zeros_like(r)
    
    for i in range(N):
        for j in range(N):
            if i != j:
                r_ij = r[j] - r[i]
                r_ij_mod3 = np.linalg.norm(r_ij)**3
                a[i] += masas[j] * r_ij / r_ij_mod3
                
    return np.concatenate((v, a.flatten()))

# %% === EJECUCIÓN Y GRÁFICOS ===


# Sistema Solar - Asumiendo nula interacción entre Júpiter y Tierra

if no_interaction:
    fig, ax = plt.subplots(figsize=(10,10), dpi=120)



    for r, vel, planeta, color in zip([r_T, r_J], [v_T, v_J], ['Tierra', 'Júpiter'], ['navy', 'darkgoldenrod']):
        sol = ja.rk4_solver(grav_2_bodies, t, [r, 0, 0, vel])
        x, y, vx, vy = sol.T
        ax.plot(x,y,label=rf'{planeta}', color = color)
        ax.plot(x[-1], y[-1], marker='o', ms=5, color=color)

    ja.setup_ax(ax, title=r'Órbitas del Sistema Solar',
                xlabel=r'$x$ [AU]', ylabel=r'$y$ [AU]', legend=True)  

    plt.plot(0, 0, marker='o', ms=15, color='gold')

    plt.tight_layout()
    plt.show()




# Sistema Solar - Incluyendo interacción Júpiter-Tierra

if w_interaction:
    fig, ax = plt.subplots(figsize=(10, 10), dpi = 120)

    # fig3D = plt.figure(figsize=(10, 10), dpi=100)
    # ax3D  = fig3D.add_subplot(projection='3d') # Hacemos un plot 3D 


    masas = np.array([GM_T, GM_J, GM_S])
    planetas = ['Tierra', 'Júpiter', 'Sol']
    colores = ['navy', 'darkgoldenrod', 'gold']

    r0 = np.array([r_T, 0.0, 0.0, r_J, 0.0, 0.0, 0.0, 0.0, 0.0])
    v0 = np.array([0.0, v_T, 0.0, 0.0, v_J, 0.0, 0.0, v_S, 0.0])

    state0 = np.concatenate((r0,v0))
    sol = ja.rk4_solver(lambda t, y: grav_n_bodies(t, y, masas), t, state0)

    n_bodies = len(masas)

    # Extraigo las posiciones + reshape (N/dt, N_bodies, 3 - x,y,z)
    r_sol = sol[:, :3*n_bodies].reshape(-1, n_bodies, 3) 


    for i in range(n_bodies):
        x = r_sol[:, i, 0]
        y = r_sol[:, i, 1]
        z = r_sol[:, i, 2]
        ax.plot(x, y, label=planetas[i], color=colores[i], alpha=0.7)
        # ax.plot(x[0], y[0], marker='o', ms=5, color=colores[i], label=f'Posición inicial {planetas[i]}')
        # ax3D.plot(x,y,z, label=planetas[i], color=colores[i], alpha=0.7)

    ja.setup_ax(ax, title=r'Órbitas del Sistema Solar ($N$ cuerpos)',
                xlabel=r'$x$ [AU]', ylabel=r'$y$ [AU]', legend=True)

    # ja.setup_ax(ax3D, title=r'Órbitas del Sistema Solar ($N$ cuerpos)',
    #             xlabel=r'$x$ [AU]', ylabel=r'$y$ [AU]',  zlabel=r'$z$ [AU]', legend=True)


    plt.tight_layout()
    plt.show()

    r_Tier = r_sol[:,0,:].T
    r_Jupi = r_sol[:,1,:].T
    r_Sol  = r_sol[:,2,:].T

    ext = 1.2
    lim_x = (np.min([r_Tier[0], r_Jupi[0], r_Sol[0]])*ext, np.max([r_Tier[0], r_Jupi[0], r_Sol[0]])*ext)
    lim_y = (np.min([r_Tier[1], r_Jupi[1], r_Sol[1]])*ext, np.max([r_Tier[1], r_Jupi[1], r_Sol[1]])*ext)
    lim_z = (np.min([r_Tier[2], r_Jupi[2], r_Sol[2]])*ext, np.max([r_Tier[2], r_Jupi[2], r_Sol[2]])*ext)
    
    ja.animar_trayectorias(
        datos=[r_Tier, r_Jupi, r_Sol], 
        duracion=5.0,        # Durará exactamente 5 segundos
        fps=30,              # A 30 fotogramas por segundo (150 frames en total)
        guardar=False, 
        archivo="figuras/sistema_solar.mp4",
        title=r"Orbitas Sistema Solar",
        xlabel=r"$x$ [m]", ylabel=r"$y$ [m]", zlabel=r"$z$ [m]",
        xlim=lim_x, ylim=lim_y, zlim=lim_z,
        colors=colores,
        labels=planetas
        )
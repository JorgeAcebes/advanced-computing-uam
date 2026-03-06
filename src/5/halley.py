# %% Imports y Declaraciones
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy as sp
from scipy.signal import find_peaks
from scipy.stats import linregress
import sys
from pathlib import Path
import importlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
import tools as ja
importlib.reload(ja)
ja.setup_style(base_size=19, dpi=120)

# === Decidir qué se grafica ===
w_interaction  = 1

# --- Variables y Funciones ---

GM_S = 4 * np.pi ** 2
GM_T = GM_S / 332946
GM_J = GM_S / 1048
GM_h = 0
GM_P = 6.6e-6 * GM_S

r_T = 1.0 # Distancia Tierra-Sol 
r_J = 5.2 # Distancia Jupiter-Sol
r_h = 0.59 # Perihelio Halley
r_P = 29.7 # Periehelio Plutón
r_S = ja.position_sum_cm([r_T, r_J, r_h], [GM_T, GM_J, GM_h])# Posición Sol


T_T = 1.0
T_J = 11.86
T_h = 76
T_P = 39.5**3/2 # Semieje mayor ** 3/2 (Kepler)


v_T = 2 * np.pi * r_T / T_T
v_J = 2 * np.pi * r_J / T_J
v_h = ja.velocity_orbit(r_h, T_h)
v_P = ja.velocity_orbit(r_P, T_P)
v_S = ja.velocity_sun_cm([v_T, v_J, v_h], [GM_T, GM_J, GM_h])

t_tot = T_h
dt = 0.0001
t = np.arange(0, t_tot, dt)

# Parámetro ficticio de Relatividad General. 
# Si es 0, recupera la física clásica.

alpha_ = 0

# -- Condiciones iniciales --

# Definimos matrices (N, 3) directamente para Verlet
r0 = np.array([
    [r_T, 0.0, 0.0],  # Tierra
    [r_J, 0.0, 0.0],  # Júpiter 
    [r_h, 0.0, 0.0],  # Halley
    # [r_P, 0.0, 0.0],  # Plutón
    [r_S, 0.0, 0.0]   # Sol 
])

v0 = np.array([
    [0.0, v_T, 0.0],
    [0.0, v_J, 0.0],
    [0.0, v_h, 0.0],
    # [0.0, v_P, 0.0],
    [0.0, v_S, 0.0]
])

# -- Funciones --

def grav_n_bodies(r, masas, alpha=0.0):
    """
    r debe tener dimensiones (N, 3). Devuelve 'a' con dimensiones (N, 3).
    """
    N = len(masas)
    a = np.zeros_like(r)
    
    for i in range(N):
        for j in range(N):
            if i != j:
                r_ij = r[j] - r[i]
                r_ij_mod = np.linalg.norm(r_ij)
                
                factor_rg = 1.0 + alpha / (r_ij_mod**2) 
                a[i] += masas[j] * r_ij / (r_ij_mod**3) * factor_rg
    return a

# %%
# Sistema Solar + Halley 

masas = np.array([GM_T, 
                  GM_J, 
                  GM_h, 
                #   GM_P, 
                  GM_S])
planetas = ['Tierra', 
            'Júpiter', 
            'Halley',
            # 'Plutón',
            'Sol']

colores = ['navy', 
           'darkgoldenrod',
           'teal',
           # 'darkviolet', 
            'gold']

n_bodies = len(masas)

r_sol, v_sol = ja.verlet_solver(lambda x: grav_n_bodies(x, masas, alpha=alpha_), t, r0, v0)



# %%
fig, ax = plt.subplots(figsize=(40, 7.5), dpi=120)

for i in range(n_bodies):
    x = r_sol[:, i, 0]
    y = r_sol[:, i, 1]
    ax.plot(x, y, label=planetas[i], color=colores[i], alpha=0.7)

ja.setup_ax(ax, title=r'Órbitas del Sistema Solar',
            xlabel=r'$x$ [AU]', ylabel=r'$y$ [AU]', legend=True)


plt.tight_layout()
plt.show()

# Extracción de datos para animación (transposición para cuadrar dimensiones)
r_Tier = r_sol[:, 0, :].T
r_Jupi = r_sol[:, 1, :].T
r_Hall = r_sol[:, 2, :].T
# r_Plut = r_sol[:, 3, :].T
r_Sol  = r_sol[:, 3, :].T

ext = 1.2
lim_x = (np.min(r_Hall[0])*ext, np.max(r_Jupi[0])*ext)
lim_y = lim_x

# ja.animar_trayectorias(
#     datos=[r_Tier, r_Jupi, r_Hall, r_Sol], 
#     duracion=5.0, fps=30, guardar=False, 
#     archivo="figuras/sistema_solar.mp4",
#     title=r"Orbitas Sistema Solar",
#     xlabel=r"$x$ [AU]", ylabel=r"$y$ [AU]", zlabel=r"$z$ [AU]",
#     xlim=lim_x,
#     ylim=lim_y, 
#     colors=colores, labels=planetas,
# )
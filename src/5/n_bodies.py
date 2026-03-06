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

r_S = 0.0 # Posición Sol
r_T = 1.0 # Distancia Tierra-Sol 
r_J = 5.2 # Distancia Jupiter-Sol

T_T = 1.0
T_J = 11.86

v_T = 2 * np.pi * r_T / T_T
v_J = 2 * np.pi * r_J / T_J
v_S = -(GM_T * v_T + GM_J * v_J) / GM_S  #Velocidad Sol para que nos situemos en el centro de masas

t_tot = 12
dt = 0.001
t = np.arange(0, t_tot, dt)

# Parámetro ficticio de Relatividad General. 
# Si es 0, recupera la física clásica.
alpha_ = 1e-3

# -- Funciones --

def grav_n_bodies(r, masas, alpha=0.0, ):
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
# Sistema Solar - N cuerpos

fig, ax = plt.subplots(figsize=(10, 10), dpi=120)

masas = np.array([GM_T, GM_J, GM_S])
planetas = ['Tierra', 'Júpiter', 'Sol']
colores = ['navy', 'darkgoldenrod', 'gold']
n_bodies = len(masas)

# Definimos matrices (N, 3) directamente para Verlet
r0 = np.array([
    [r_T, 0.0, 0.0],  # Tierra
    [r_J, 0.0, 0.0],  # Júpiter 
    [0.0, 0.0, 0.0]   # Sol 
])

v0 = np.array([
    [0.0, v_T, 0.0],
    [0.0, v_J, 0.0],
    [0.0, v_S, 0.0]
])

r_sol, v_sol = ja.verlet_solver(lambda x: grav_n_bodies(x, masas, alpha=alpha_), t, r0, v0)

for i in range(n_bodies):
    x = r_sol[:, i, 0]
    y = r_sol[:, i, 1]
    ax.plot(x, y, label=planetas[i], color=colores[i], alpha=0.7)

ja.setup_ax(ax, title=r'Órbitas del Sistema Solar ($N$ cuerpos)',
            xlabel=r'$x$ [AU]', ylabel=r'$y$ [AU]', legend=True)

plt.tight_layout()
plt.show()

# Extracción de datos para animación (transposición para cuadrar dimensiones)
r_Tier = r_sol[:, 0, :].T
r_Jupi = r_sol[:, 1, :].T
r_Sol  = r_sol[:, 2, :].T

ext = 1.2
lim_x = (np.min([r_Tier[0], r_Jupi[0], r_Sol[0]])*ext, np.max([r_Tier[0], r_Jupi[0], r_Sol[0]])*ext)
lim_y = (np.min([r_Tier[1], r_Jupi[1], r_Sol[1]])*ext, np.max([r_Tier[1], r_Jupi[1], r_Sol[1]])*ext)
lim_z = (np.min([r_Tier[2], r_Jupi[2], r_Sol[2]])*ext, np.max([r_Tier[2], r_Jupi[2], r_Sol[2]])*ext)

ja.animar_trayectorias(
    datos=[r_Tier, r_Jupi, r_Sol], 
    duracion=5.0, fps=30, guardar=False, 
    archivo="figuras/sistema_solar.mp4",
    title=r"Orbitas Sistema Solar",
    xlabel=r"$x$ [m]", ylabel=r"$y$ [m]", zlabel=r"$z$ [m]",
    xlim=lim_x, ylim=lim_y, zlim=lim_z,
    colors=colores, labels=planetas
)


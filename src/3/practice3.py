# %% === IMPORTS y DEFINICIONES 
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
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
def formato_lanzamiento(ax):
    ax.set_xlabel(r'Alcance $x$ [m]')
    ax.set_ylabel(r'Altura $y$ [m]')
    # Detalles finos (Grilla y Ticks)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.8) # Eje cero
    ax.grid(True, which='major', linestyle='-', alpha=0.6, color='#d3d3d3')
    ax.minorticks_on() # Ticks menores son esenciales para lectura precisa
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)

    # Leyenda (para eso sirve label)
    ax.legend(frameon=True, loc='best', fontsize=12, framealpha=0.6, edgecolor='gray')



# ==========================================================================================

#________________ PARÁMETROS Y FUNCIONES _______________________


# --- Definimos los parámetros que vamos a emplear
v0 = 70 # [m/s] Velocidad inicial en módulo. 
p0 = 1.225 # [kg/m**3] Densidad del aire a nivel del mar
g = sp.constants.g
RT = 6371000 #[m] Radio terrestre
a = 6.5e-3 #Valor de a en la fórmula adiabática
alpha = 2.5 # Exponente en la fórmula adibática
T0 = 293 #[K] Temperatura ambiente, para adibática
S0wm = 0.25 # [1/m]
R_golf = 0.02 # [m] Radio pelota de golf
A_golf = np.pi * R_golf**2

m_golf = 0.045 #[kg] Masa pelota de golf

# --- Valores iniciales e intervalo de tiempo ---

x0, y0, z0 = 0, 0, 0
vwx = 10 # [m/s] Velocidad del viento en la dirección x

t = np.linspace(0,1000,10000)


def magnus_3D(t, vec_ini):

    x, y, z = vec_ini[0], vec_ini[1], vec_ini[2]
    vx, vy, vz = vec_ini[3], vec_ini[4], vec_ini[5]

#   Velocidades
    dx = vx 
    dx_rel = dx - vwx #velocidad real (quitandole la componente del viento)
    dy = vy
    dz = vz
    
    v = np.sqrt(dx_rel**2 + dy**2 + dz**2) 

    if v < 14:
        C = 1.0
    else:
        C = 14 / v
    
    rho = (1 - (a * z / T0)) ** alpha

    arrastre = C * rho * A_golf / m_golf

    dvx = - arrastre * v * dx_rel - S0wm * dy
    dvy = - arrastre * v * dy     + S0wm * dx_rel
    dvz = -g *(RT / (RT + z)) ** 2 - arrastre * v * dz 

    return np.array([dx, dy, dz, dvx, dvy, dvz])

def magnus_3D_hooke(t, vec_ini):
    x, y, z = vec_ini[0], vec_ini[1], vec_ini[2]
    vx, vy, vz = vec_ini[3], vec_ini[4], vec_ini[5]

#   Velocidades
    dx = vx 
    dx_rel = dx - vwx #velocidad real (quitandole la componente del viento)
    dy = vy
    dz = vz
    
    v = np.sqrt(dx_rel**2 + dy**2 + dz**2) 

    if v < 14:
        C = 1.0
    else:
        C = 14 / v
    
    rho = (1 - (a * z / T0)) ** alpha

    arrastre = C * rho * A_golf / m_golf

    dvx = -arrastre * v * dx_rel + S0wm * dy
    dvy = -arrastre * v * dy     - S0wm * dx_rel
    dvz = -g *(RT / (RT + z)) ** 2 - arrastre * v * dz 

    return np.array([dx, dy, dz, dvx, dvy, dvz])



# %% === Magnus en 3D ===

theta = np.radians(45)  #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
vx0 = v0 * np.cos(theta)
vy0 = 0
vz0 = v0 * np.sin(theta)

inicial_state = [x0,y0,z0,vx0,vy0,vz0]

sol_magnus_3D = ja.euler_solver(magnus_3D, t, inicial_state, 2)
mag_3D = sol_magnus_3D.T
t_magnus_3D = t[:len(mag_3D[2])]

sol_magnus_3D_neg = ja.euler_solver(magnus_3D_hooke, t, inicial_state, 2)
mag_3D_neg = sol_magnus_3D_neg.T
t_magnus_3D_neg = t[:len(mag_3D_neg[2])]

# %%

vx = mag_3D_neg[3]
vy = mag_3D_neg[4]
vz = mag_3D_neg[5]

v = np.sqrt(vx **2 + vy**2 + vz**2)

# %% === Magnus en 3D (Animado) ===

import matplotlib.animation as animation

fig = plt.figure(figsize=(10, 8), dpi=100)
ax = fig.add_subplot(projection='3d')

line, = ax.plot([], [], [], lw=2, color='violet', label='Trayectoria') 
point, = ax.plot([], [], [], 'o', color='crimson', markersize=5) # El punto que está marcando el movimiento
shadow, = ax.plot([], [], [], '--', color='gray', alpha=0.5) # Proyección en el plano del suelo


line_h, = ax.plot([], [], [], lw=2, color='green', label='Trayectoria') 
point_h, = ax.plot([], [], [], 'o', color='lime', markersize=5) # El punto que está marcando el movimiento
shadow_h, = ax.plot([], [], [], '--', color='gray', alpha=0.5) # Proyección en el plano del suelo


ax.set_xlabel('$x$ [m]', labelpad = 15)
ax.set_ylabel('$y$ [m]', labelpad = 15)
ax.set_zlabel('$z$ [m]', labelpad = 15)


x_vals = mag_3D[0]
y_vals = mag_3D[1]
z_vals = mag_3D[2]

x_hook = mag_3D_neg[0]
y_hook = mag_3D_neg[1]
z_hook = mag_3D_neg[2]

ax.set_xlim(min(np.min(x_vals), np.min(x_hook)), max(np.max(x_vals), np.max(x_hook)))
ax.set_ylim(min(np.min(y_vals), np.min(y_hook)), max(np.max(y_vals), np.max(y_hook)))
ax.set_zlim(min(np.min(z_vals), np.min(z_hook)), max(np.max(z_vals), np.max(z_hook)))


# Para limitar el número de números que aparecen en los ejes
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax.zaxis.set_major_locator(MaxNLocator(nbins=5))

frames_totales = 400 
step = max(1, max(len(x_vals), len(x_hook)) // frames_totales)

def update(frame):
    # El índice real en el array de datos
    idx = frame * step
    if idx >= len(x_vals):
        idx = len(x_vals) - 1
        
    line.set_data(x_vals[:idx], y_vals[:idx])
    line.set_3d_properties(z_vals[:idx])
    
    point.set_data([x_vals[idx]], [y_vals[idx]]) 
    point.set_3d_properties([z_vals[idx]])

    shadow.set_data(x_vals[:idx], y_vals[:idx])
    shadow.set_3d_properties(np.zeros(idx))


    idx_hook = frame * step
    if idx_hook >= len(x_hook):
        idx_hook = len(x_hook)-1
    
    line_h.set_data(x_hook[:idx_hook], y_hook[:idx_hook])
    line_h.set_3d_properties(z_hook[:idx_hook])
    
    point_h.set_data([x_hook[idx_hook]], [y_hook[idx_hook]]) 
    point_h.set_3d_properties([z_hook[idx_hook]])

    shadow_h.set_data(x_hook[:idx_hook], y_hook[:idx_hook])
    shadow_h.set_3d_properties(np.zeros(idx_hook))

    
    return line, point, shadow, line_h, point_h, shadow_h

ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=frames_totales, 
    interval=2, # milisegundos entre frames
    blit=False   # blit=False suele ser más estable en 3D
)




plt.show()
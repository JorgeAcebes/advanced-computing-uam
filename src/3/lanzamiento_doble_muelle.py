# %% === IMPORTS Y DEFINICIONES
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



# %% 2 PELOTAS
# --- Valores Físicos ---

m1, m2, m3 = 1, 2, 3 # [kg] Masas de los cuerpos
L = 1 # [m] Longitud de la muelle (todos los muelles iguales)
k = 100 # [N/m] constante elástica muelle
c = 30.1 # [N * s / m] constante viscosa del fluido
g = np.array([0,0, -sp.constants.g]) #Constante gravitatoria en formato vector (vamos a trabajar con vectores)


# --- Ecuación de movimiento ---

def muelle_viscoelástico(t, estado): # Implementación del modelo de Kelvin-Voig
    r1 = estado[0:3]
    v1 = estado[3:6]
    r2 = estado[6:9]
    v2 = estado[9:12]

    r12 = r1 - r2
    d12 = np.linalg.norm(r12) # Modulo de la distancia relativa
    u12 = r12 / (d12 + 1e-12) # Vector unitario en la dirección 1-2. Le sumo +10^-15 para evitar divisiones por 0

    v12 = v1 - v2

    F_viscelast =  - ( k * (d12 - L)  + c * np.dot(v12, u12) ) * u12
    a1 =  g/m1 + F_viscelast / m1 
    a2 =  g/m2 - F_viscelast / m2

    return np.concatenate((v1,a1,v2,a2))

y0 = [-10, 10, 0,       # r1
      0, 0, 10,       # v1
      10, 0, 0,      # r2
      -20, -10, 20]       # v2


t = np.linspace(0,10, 10000)



sol_visc_elas_all = ja.rk4_solver(muelle_viscoelástico, t, y0, 2)

sol_visc_elas = sol_visc_elas_all.T


R1 = sol_visc_elas[0:3]
V1 = sol_visc_elas[3:6]
R2 = sol_visc_elas[6:9]
V2 = sol_visc_elas[9:12]


x1, y1, z1 = R1[0:3]
x2, y2, z2 = R2[0:3]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')


# Elementos gráficos a actualizar
line1, = ax.plot([], [], [], 'b--', lw=1, alpha=0.5, label='Trayectoria 1')
point1, = ax.plot([], [], [], 'bo', ms=8)
line2, = ax.plot([], [], [], 'r--', lw=1, alpha=0.5, label='Trayectoria 2')
point2, = ax.plot([], [], [], 'ro', ms=8)
muelle, = ax.plot([], [], [], 'k-', lw=2)

# Configuración de ejes FIJA 
ax.set_xlim(min(np.min(x1), np.min(x2)), max(np.max(x1), np.max(x2)))
ax.set_ylim(min(np.min(y1), np.min(y2)), max(np.max(y1), np.max(y2)))
ax.set_zlim(0, max(np.max(z1), np.max(z2)) + 1)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title(f'2 Cuerpos unidos por muelle (k={k}, c={c})')
ax.legend()

def update(frame):
    line1.set_data(x1[:frame], y1[:frame])
    line1.set_3d_properties(z1[:frame])
    
    line2.set_data(x2[:frame], y2[:frame])
    line2.set_3d_properties(z2[:frame])
    
    # Actualizar posiciones actuales
    point1.set_data([x1[frame]], [y1[frame]]) 
    point1.set_3d_properties([z1[frame]])
    
    point2.set_data([x2[frame]], [y2[frame]])
    point2.set_3d_properties([z2[frame]])
    
    # Actualizar muelle 
    muelle.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
    muelle.set_3d_properties([z1[frame], z2[frame]])
    
    return line1, point1, line2, point2, muelle

# Crear animación
ani = FuncAnimation(fig, update, frames=range(0,len(t), 20), # Con esto puedo modificar rapidez del vídeo
                     interval=10, blit=False)

plt.show()


# %% 3 cuerpos

# --- Ecuación de movimiento ---

k2 = k/50.0
k3 = k*50

def muelle_viscoelástico(t, estado): # Implementación del modelo de Kelvin-Voig
    r1 = estado[0:3]
    v1 = estado[3:6]
    r2 = estado[6:9]
    v2 = estado[9:12]
    r3 = estado[12:15]
    v3 = estado[15:18]

    r12 = r1 - r2
    d12 = np.linalg.norm(r12) # Modulo de la distancia relativa
    u12 = r12 / (d12 + 1e-12) # Vector unitario en la dirección 1-2. Le sumo +10^-15 para evitar divisiones por 0
    v12 = v1 - v2

    r13 = r1 - r3
    d13 = np.linalg.norm(r13)
    u13 = r13/ (d13 + 1e-12)
    v13 = v1 - v3

    r23 = r2 - r3
    d23 = np.linalg.norm(r23)
    u23 = r23 / (d23 + 1e-12)
    v23 = v2 - v3

    F_viscelast_12 =  - ( k * (d12 - L)  + c * np.dot(v12, u12) ) * u12
    F_viscelast_13 =  - ( k2 * (d13 - L)  + c * np.dot(v13, u13) ) * u13
    F_viscelast_23 =  - ( k3 * (d23 - L)  + c * np.dot(v23, u23) ) * u23

    a1 =  g/m1 + (F_viscelast_12 + F_viscelast_13) / m1 
    a2 =  g/m2 + (F_viscelast_23 - F_viscelast_12) / m2
    a3 =  g/m3 - (F_viscelast_13 + F_viscelast_23) / m3

    return np.concatenate((v1,a1,v2,a2,v3,a3))

y0 = [-1, 1, 1,   # r1
      2, 0, 5,       # v1
      0, 2, 2,       # r2
     -2, -1, 2,   # v2
     0, 10, 0,         # r3
      2, 0, 2        # v3
    ]


t = np.linspace(0,20, 10000)



sol_visc_elas_all = ja.rk4_solver(muelle_viscoelástico, t, y0)

sol_visc_elas = sol_visc_elas_all.T


R1 = sol_visc_elas[0:3]
V1 = sol_visc_elas[3:6]
R2 = sol_visc_elas[6:9]
V2 = sol_visc_elas[9:12]
R3 = sol_visc_elas[12:15]
V3 = sol_visc_elas[15:18]


x1, y1, z1 = R1[0:3]
x2, y2, z2 = R2[0:3]
x3, y3, z3 = R3[0:3]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')


# Elementos gráficos a actualizar
line1, = ax.plot([], [], [], 'b--', lw=1, alpha=0.5, label='Trayectoria 1')
point1, = ax.plot([], [], [], 'bo', ms=8)
line2, = ax.plot([], [], [], 'r--', lw=1, alpha=0.5, label='Trayectoria 2')
point2, = ax.plot([], [], [], 'ro', ms=8)
line3, = ax.plot([], [], [], 'g--', lw=1, alpha=0.5, label='Trayectoria 3')
point3, = ax.plot([], [], [], 'go', ms=8)
muelle12, = ax.plot([], [], [], 'k-', lw=2) 
muelle13, = ax.plot([], [], [], 'k-', lw=2) 
muelle23, = ax.plot([], [], [], 'k-', lw=2) 

# Configuración de ejes FIJA 
ax.set_xlim(min(np.min(x1), np.min(x2), np.min(x3)), max(np.max(x1), np.max(x2), np.max(x3)))
ax.set_ylim(min(np.min(y1), np.min(y2), np.min(y3)), max(np.max(y1), np.max(y2), np.max(y3)))
ax.set_zlim(min(np.min(z1), np.min(z2), np.min(z3)), max(np.max(z1), np.max(z2), np.max(z3)) + 1)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title(f'3 Cuerpos unidos por muelles (k={k}, c={c})')
ax.legend()

def update(frame):
    line1.set_data(x1[:frame], y1[:frame])
    line1.set_3d_properties(z1[:frame])
    
    line2.set_data(x2[:frame], y2[:frame])
    line2.set_3d_properties(z2[:frame])
    
    line3.set_data(x3[:frame], y3[:frame])
    line3.set_3d_properties(z3[:frame])
    
    # Actualizar posiciones actuales
    point1.set_data([x1[frame]], [y1[frame]]) 
    point1.set_3d_properties([z1[frame]])
    
    point2.set_data([x2[frame]], [y2[frame]])
    point2.set_3d_properties([z2[frame]])
    
    point3.set_data([x3[frame]], [y3[frame]])
    point3.set_3d_properties([z3[frame]])
    
    # Actualizar muelles
    muelle12.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
    muelle12.set_3d_properties([z1[frame], z2[frame]])
    
    muelle13.set_data([x1[frame], x3[frame]], [y1[frame], y3[frame]])
    muelle13.set_3d_properties([z1[frame], z3[frame]])

    muelle23.set_data([x2[frame], x3[frame]], [y2[frame], y3[frame]])
    muelle23.set_3d_properties([z2[frame], z3[frame]])

    return line1, point1, line2, point2, line3, point3, muelle12, muelle13, muelle23

# Crear animación
ani = FuncAnimation(fig, update, frames=range(0,len(t), 20), # Con esto puedo controlar rapidez animación
                     interval=10, blit=False)

# ani.save('muelle_fisica.mp4', fps=30)
plt.show()
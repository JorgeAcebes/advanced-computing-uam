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
def espacio_fases(ax):
    ax.set_xlabel(r'Posición $x$ [m]')
    ax.set_ylabel(r'Velocidad $v$ [m/s]')
    # Detalles finos (Grilla y Ticks)
    ax.grid(True, which='major', linestyle='-', alpha=0.6, color='#d3d3d3')
    # Leyenda (para eso sirve label)
    ax.legend(frameon=True, loc='best', fontsize=12, framealpha=0.6, edgecolor='gray')







# --- Configuración péndulos ---
m1, m2 = 1.0, 2.0 # kg
L1, L2  = 3.0, 2.0 # Longitud cuerda [m]
g = sp.constants.g

def pend_doble(t, state):
    o1, w1 = state[0], state[1]
    o2, w2 = state[2], state[3]

    a1 = (- g * (2*m1 + m2) * np.sin(o1) - m2 * g * np.sin(o1 - 2*o2) - 2 * np.sin(o1-o2) * m2 * (w2**2 * L2 + w1**2 * L1 * np.cos(o1-o2))) / (L1 * (2 * m1 + m2 - m2 * np.cos(2*o1 -2*o2)))
    a2 =(2* np.sin(o1-o2) * (w1**2 *L1 * (m1+m2) + g*(m1 + m2) * np.cos(o1) + w2**2 *L2 *m2 * np.cos(o1-o2))) / (L2 * (2 *m1 +m2 - m2 * np.cos(2*o1-2*o2)))

    return np.array([w1, a1, w2, a2])


# %%

y0 = [float(np.pi), 0.0, float(np.pi-1e-12), 0.0]

N = 100

t = np.linspace(0,N, N*100)

sol_dp = ja.rk4_solver(pend_doble, t, y0)

sol = sol_dp.T

O1, O2 = sol[0], sol[2]
W1, W2 = sol[1], sol[3]


x1 = L1 * np.sin(O1)
y1 =-L1 * np.cos(O1)

x2 = x1 + L2 * np.sin(O2)
y2 = y1 - L2 * np.cos(O2)

R1 = np.sqrt(x1**2 + y1**2)
R2 = np.sqrt(x2**2 + y2**2)

V1 = W1 * L1
V2 = W2 * L2



fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

plt.plot(V2, R2, ':', alpha = 0.5, color = 'violet', label='Bola 2', zorder =1)

espacio_fases(ax)

ax.set_title(r'Espacio de fases Bola 2')

# Guardado y Muestra
plt.tight_layout()
# plt.savefig('plot_fisica.pdf', bbox_inches='tight') # Formato vectorial preferido
plt.show()
# plt.close()


# %%


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()

# Elementos gráficos a actualizar
cuerda, = ax.plot([], [], 'k-', lw=2, alpha = 0.3) 
cuerda_centro, = ax.plot([], [], 'k-', lw =2, alpha = 0.3)
point_centro, = ax.plot([], [], 'ko', ms=8, zorder = 10)
line1, = ax.plot([], [], 'g--', lw=1, alpha=0.2, label='Trayectoria 1')
point1, = ax.plot([], [], 'go', ms=8, zorder = 10)
line2, = ax.plot([], [], '--', color='violet', lw=1, alpha=0.2, label='Trayectoria 2')
point2, = ax.plot([], [], 'o', color = 'violet', ms=8, zorder = 10)



ax.set_xlim(min(np.min(x1), np.min(x2), np.min(y1), np.min(y2)), max(np.max(x1), np.max(x2), np.max(y1), np.max(y2)))
ax.set_ylim(min(np.min(x1), np.min(x2), np.min(y1), np.min(y2)), max(np.max(x1), np.max(x2), np.max(y1), np.max(y2)))
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.legend()

def update(frame):
    # Actualizar cuerda (línea entre bolas)
    cuerda.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
    cuerda_centro.set_data([0, x1[frame]], [0, y1[frame]])

    # Actualizar estelas (historia hasta el frame actual)

    line1.set_data(x1[:frame], y1[:frame])
    
    line2.set_data(x2[:frame], y2[:frame])
    
    # Actualizar posiciones actuales (bolas)
    point1.set_data([x1[frame]], [y1[frame]]) # Requiere lista o array
    
    point2.set_data([x2[frame]], [y2[frame]])

    point_centro.set_data([0],[0])
    


    return line1, point1, line2, point2, cuerda, point_centro, cuerda_centro

# Crear animación
ani = FuncAnimation(fig, update, frames=range(0,len(t), 10), interval=20, blit=False)

# Para guardar: ani.save('cuerda_fisica.mp4', fps=30)
plt.show()
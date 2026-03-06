import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sys

# Parámetro de masa (Tierra-Luna)
mu = 0.0121505856

# --- Cálculo riguroso de los Puntos de Lagrange ---
def eq_colineales(x, mu):
    return x - (1 - mu) * np.sign(x + mu) / (x + mu)**2 - mu * np.sign(x - 1 + mu) / (x - 1 + mu)**2

# Estimaciones iniciales basadas en desarrollo de Hill
guess_L1 = 1 - (mu/3)**(1/3)
guess_L2 = 1 + (mu/3)**(1/3)
guess_L3 = -(1 + 5*mu/12)

x_L1 = fsolve(eq_colineales, guess_L1, args=(mu,))[0]
x_L2 = fsolve(eq_colineales, guess_L2, args=(mu,))[0]
x_L3 = fsolve(eq_colineales, guess_L3, args=(mu,))[0]

L_points = {
    1: (x_L1, 0.0, "L1 (Inestable)"),
    2: (x_L2, 0.0, "L2 (Inestable)"),
    3: (x_L3, 0.0, "L3 (Inestable)"),
    4: (0.5 - mu, np.sqrt(3)/2, "L4 (Estable)"),
    5: (0.5 - mu, -np.sqrt(3)/2, "L5 (Estable)")
}

# --- Interacción ---
try:
    eleccion = int(input("Introduce el punto de Lagrange a simular (1, 2, 3, 4 o 5): "))
    if eleccion not in L_points:
        raise ValueError
except ValueError:
    print("Entrada inválida. Ejecuta de nuevo e introduce un entero del 1 al 5.")
    sys.exit()

x_L, y_L, nombre_L = L_points[eleccion]

# --- Dinámica CR3BP ---
def derivadas_cr3bp(t, estado, mu):
    x, y, z, vx, vy, vz = estado
    r1_3 = ((x + mu)**2 + y**2 + z**2)**1.5
    r2_3 = ((x - 1 + mu)**2 + y**2 + z**2)**1.5
    
    dU_dx = x - ((1 - mu) * (x + mu) / r1_3) - (mu * (x - 1 + mu) / r2_3)
    dU_dy = y - ((1 - mu) * y / r1_3) - (mu * y / r2_3)
    dU_dz = -((1 - mu) * z / r1_3) - (mu * z / r2_3)
    
    return [vx, vy, vz, 2 * vy + dU_dx, -2 * vx + dU_dy, dU_dz]

# Condición inicial: Perturbación espacial de 0.01 en X
estado_inicial = [x_L + 0.01, y_L, 0.0, 0.0, 0.0, 0.0]

# Ajuste del tiempo: Puntos inestables divergen rápido.
t_final = 20 if eleccion in [1, 2, 3] else 150
frames = 1000 if eleccion in [1, 2, 3] else 3000

t_span = (0, t_final)
t_eval = np.linspace(t_span[0], t_span[1], frames)

solucion = solve_ivp(
    fun=derivadas_cr3bp, t_span=t_span, y0=estado_inicial, args=(mu,), 
    t_eval=t_eval, method='DOP853', rtol=1e-11, atol=1e-13
)

x_orb, y_orb = solucion.y[0], solucion.y[1]

# --- Animación ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')

# Margen dinámico basado en la órbita resultante
margen = 0.2
ax.set_xlim(np.min(x_orb) - margen, np.max(x_orb) + margen)
ax.set_ylim(np.min(y_orb) - margen, np.max(y_orb) + margen)

ax.scatter([-mu, 1-mu], [0, 0], color=['blue', 'gray'], s=[100, 30], zorder=3, label='Primarios')
ax.scatter([x_L], [y_L], color='red', marker='x', zorder=3, label=f'Punto {nombre_L}')

linea_estela, = ax.plot([], [], color='black', linewidth=0.8, alpha=0.6)
punto_particula, = ax.plot([], [], 'ko', markersize=5, zorder=4)

def init():
    linea_estela.set_data([], [])
    punto_particula.set_data([], [])
    return linea_estela, punto_particula

def update(frame):
    linea_estela.set_data(x_orb[:frame], y_orb[:frame])
    punto_particula.set_data([x_orb[frame]], [y_orb[frame]])
    return linea_estela, punto_particula

ani = animation.FuncAnimation(
    fig, update, frames=len(t_eval), init_func=init, 
    blit=True, interval=10, repeat=False
)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title(f'Perturbación en {nombre_L}')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
plt.show()
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


figuras = Path(__file__).resolve().parent.parent / '4' / 'figures'
figuras.mkdir(parents=True, exist_ok=True)  



plot_animation = 1 # Decide si mostrar o no la animación. En caso de mostrar la animación, no se muestra la gráfica de análisis.


# --- Variables y Funciones ---

sigma = 10.0
b     = 8/3

def lorenz_atractor(t,state, r):
    x, y, z = state[0], state[1], state[2]

    # Ecuaciones atractor de Lorenz
    vx = sigma * (y - x)
    vy = r * x - y - x * z
    vz = x * y - b * z

    return np.array([vx, vy, vz])


def func_update(frame):
    # El índice real en el array de datos
    idx = frame * step
    if idx >= len(x_vals):
        idx = len(x_vals) - 1
        
    line.set_data(x_vals[:idx], y_vals[:idx])
    line.set_3d_properties(z_vals[:idx])
    
    point.set_data([x_vals[idx]], [y_vals[idx]]) 
    point.set_3d_properties([z_vals[idx]])

    shadow.set_data(x_vals[:idx], y_vals[:idx])
    shadow.set_3d_properties(np.dot(np.ones(idx), np.min(z_vals)))
    
    return line, point, shadow


t_tot = 100
dt = 0.01

# Distintas r's para estudiar
R_list = [
        28,
        # 100,
        # 150, 
        # 160,
        # 170,
        # 200,
        # 300,
        # 400,
           ]

L= len(R_list)

r_anim = R_list # Distintas r's pertenecientes a R_list para las que queremos animar


# Condiciones inciales
vec_1 = [1, 0, 0]
vec_2 = [1.1, 0, 0]

# Inicialización de valores para animar
x_vals, y_vals, z_vals = 0, 0, 0

# %% ANÁLISIS EXTENSO CAOS

fig, ax = plt.subplots(L, 3, figsize=(48, 7*L), dpi=120)
if plot_animation == 1: 
    plt.close()

for i, R in enumerate(R_list):

    t = np.arange(0,t_tot,dt) #Necesario porque modifico t si t>100

    # Resolvemos las EDOS
    sol_1 = ja.rk4_solver(lambda t, y: lorenz_atractor(t, y, R), t, vec_1)
    x1, y1, z1  = sol_1.T

    sol_2 = ja.rk4_solver(lambda t, y: lorenz_atractor(t, y, R), t, vec_2)
    x2, y2, z2 = sol_2.T

    r1 = np.linalg.norm(np.array([x1, y1, z1]), axis=0)
    r2 = np.sqrt(x2**2 + y2**2 + z2**2 )


    if plot_animation:
        if any((r_ani - R) < 1e-2 for r_ani in r_anim):
            x_vals = x1
            y_vals = y1
            z_vals = z1
            import matplotlib.animation as animation
            from matplotlib.ticker import MaxNLocator

            fig = plt.figure(figsize=(10, 8), dpi=100)
            ax_anim = fig.add_subplot(projection='3d')

            line, = ax_anim.plot([], [], [], lw=2, color='violet', label='Trayectoria') 
            point, = ax_anim.plot([], [], [], 'o', color='crimson', markersize=5) # El punto que está marcando el movimiento
            shadow, = ax_anim.plot([], [], [], '--', color='gray', alpha=0.5) # Proyección en el plano del suelo


            ax_anim.set_xlabel('$x$ [m]', labelpad = 15)
            ax_anim.set_ylabel('$y$ [m]', labelpad = 15)
            ax_anim.set_zlabel('$z$ [m]', labelpad = 15)


            extra_factor = 1.2
            ax_anim.set_xlim(np.min(x_vals)*extra_factor, np.max(x_vals)*extra_factor)
            ax_anim.set_ylim(np.min(y_vals)*extra_factor, np.max(y_vals)*extra_factor)
            ax_anim.set_zlim(np.min(z_vals)*extra_factor, np.max(z_vals)*extra_factor)


            # Para limitar el número de números que aparecen en los ejes
            ax_anim.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax_anim.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax_anim.zaxis.set_major_locator(MaxNLocator(nbins=5))


            # ax.set_box_aspect((2, 1, 2)) # Escojo un tamaño de box particular (eje z más estrecho)
            ax_anim.set_title(rf'Atractor de Lorenz  $[r = {R}]$')


            frames_totales = len(x_vals)//20
            step = max(1, len(x_vals) // frames_totales)


            ani = animation.FuncAnimation(
                fig, 
                func_update, 
                frames=frames_totales, 
                interval=1, # milisegundos entre frames
                blit=False,   # blit=False suele ser más estable en 3D
                repeat = False
            )

            # Para guardar la animación. 
            ani.save(figuras/'lorenz.mp4', fps=30)
            plt.show()

     
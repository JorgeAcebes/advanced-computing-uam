# %% Imports y Declaraciones
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy as sp
from scipy.stats import linregress
from scipy.optimize import fsolve
import sys
from pathlib import Path
import importlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

figuras = Path(__file__).resolve().parent.parent / '5' / 'figures'
figuras.mkdir(parents=True, exist_ok=True)  

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
import tools as ja
importlib.reload(ja)
ja.setup_style(base_size=12, dpi=120)

# Decide que ejecutar
trayectorias = 1
pozo_pot = 1


# Decide si guardar/animar o no
save_figs = 0
animate = 0

# --- Variables y Funciones ---
mu = 0.01215  # Masa Luna / (Masa Luna + Masa Tierra)
x1, x2 = -mu, 1 - mu
t_array = np.linspace(0, 30, 3000)


# Funciones para representar el pozo de potencial gravitatorio + movimiento
def r1_3d(x, y, z):
    return np.sqrt((x - x1)**2 + y**2 + z**2)

def r2_3d(x, y, z):
    return np.sqrt((x - x2)**2 + y**2 + z**2)

def orbit_lagrange(t, estado):
    x, y, z, vx, vy, vz = estado
    
    r_13 = r1_3d(x, y, z)**3
    r_23 = r2_3d(x, y, z)**3
    term_1 = (1 - mu) / r_13
    term_2 = mu / r_23
    
    ax = 2*vy + x - term_1*(x - x1) - term_2*(x - x2)
    ay = -2*vx + y - term_1*y - term_2*y
    az = - term_1*z - term_2*z
    
    return np.array([vx, vy, vz, ax, ay, az])

def U_eff(x, y): # Pozo de potencial
    r1 = np.sqrt((x - x1)**2 + y**2)
    r2 = np.sqrt((x - x2)**2 + y**2)
    return -(1 - mu) / r1 - mu / r2 - 0.5 * (x**2 + y**2)

def grad_U_x(x):
    '''
    Gradiente del potencial en la dirección de x (ya que L4 y L5 están en posiciones resolubles analíticamente, y L1, L2, L3 están en la línea que une Tierra - Sol)
    '''
    r13 = abs(x - x1)**3
    r23 = abs(x - x2)**3
    return (1 - mu)*(x - x1)/r13 + mu*(x - x2)/r23 - x


# === Minimizamos el gradiente para obtener los puntos L1, L2, L3

x_L1 = fsolve(grad_U_x, x2 - (mu/3)**(1/3))[0]
x_L2 = fsolve(grad_U_x, x2 + (mu/3)**(1/3))[0]
x_L3 = fsolve(grad_U_x, -1.0)[0]

x_L4, y_L4 = 0.5 - mu, np.sqrt(3)/2
x_L5, y_L5 = 0.5 - mu, -np.sqrt(3)/2

pts_lagrange = {
    'L1': (x_L1, 0.0), 'L2': (x_L2, 0.0), 'L3': (x_L3, 0.0), 'L4': (x_L4, y_L4), 'L5': (x_L5, y_L5)
}

print("Coordenadas L_i:")
for k, v in pts_lagrange.items():
    print(f"{k}: x = {v[0]:.5f}, y = {v[1]:.5f}")


# Dos perturbaciones Delta[x,y,z,vx,vy,vz]

# Desplazamiento z
pert_1 = np.array([1e-3, 0.0, 2e-3, 0.0, 0.0, 0.0]) 

# Boost transversal + z
pert_2 = np.array([0.0, 1e-3, -1e-3, 1e-4, -1e-4, 2e-4]) 


# --- Trayectorias ---
if trayectorias:

    fig1 = plt.figure(figsize=(14, 10))
    fig1.canvas.manager.set_window_title('Dinámica 3D - Puntos de Lagrange')
    
    pts_evaluar = {k: pts_lagrange[k] for k in ['L1', 'L2', 'L3', 'L4']}
    
    for nombre, (x0, y0) in pts_evaluar.items():
        estado0_1 = np.array([x0, y0, 0.0, 0.0, 0.0, 0.0]) + pert_1
        estado0_2 = np.array([x0, y0, 0.0, 0.0, 0.0, 0.0]) + pert_2
        
        sol1 = ja.rk4_solver(orbit_lagrange, t_array, estado0_1)
        sol2 = ja.rk4_solver(orbit_lagrange, t_array, estado0_2)
        
        fig = plt.figure(figsize=(10, 8))
        fig.canvas.manager.set_window_title(f'Dinámica 3D - {nombre}')
        ax = fig.add_subplot(111, projection='3d')
        
        # Gráfico Principal
        ax.plot(sol1[:,0], sol1[:,1], sol1[:,2], color='violet', lw=1.2, label=r'$\delta_1$')
        ax.plot(sol2[:,0], sol2[:,1], sol2[:,2], color='lawngreen', lw=1.2, label=r'$\delta_2$')
        ax.scatter(x0, y0, 0, color='black', marker='3', s=60, label=nombre)

        ax.plot(0,0, label = 'Tierra', marker = 'o', ms = 10, color = 'green')
        ax.plot(1,0, label = 'Luna', marker = 'o', ms = 3, color = 'gold')


        ax.set_title(rf'Evolución alrededor de ${nombre}$')
        ax.set_xlabel(r'$x [NU]$')
        ax.set_ylabel(r'$y [NU]$')
        ax.set_zlabel(r'$z [NU]$')
        
        ax.set_xlim(x0 - 2, x0 + 0.5)
        ax.set_ylim(y0 - 1, y0 + 1)
        ax.set_zlim(-1, 1)
        ax.legend(fontsize=9, loc='upper left')

        # Inset
        axin = ax.inset_axes([0.65, 0.65, 0.3, 0.3], projection='3d')
        
        axin.plot(sol1[:,0], sol1[:,1], sol1[:,2], color='violet', lw=1.2)
        axin.plot(sol2[:,0], sol2[:,1], sol2[:,2], color='lawngreen', lw=1.2)
        axin.scatter(1,0,0, color = 'gold', lw = 3, s=10)
        axin.scatter(x0, y0, 0, color='black', marker='3', s=40)
        
        # Límites 
        if nombre == 'L1':
            delta = 0.15
        elif nombre == 'L2':
            delta = 0.18
        elif nombre == 'L3':
            delta = 0.05
        else:
            delta = 0.01


        axin.set_xlim(x0 - delta, x0 + delta)
        axin.set_ylim(y0 - delta, y0 + delta)
        axin.set_zlim(-delta, delta)
        
        axin.set_xticklabels([])
        axin.set_yticklabels([])
        axin.set_zticklabels([])

        if save_figs:
            plt.savefig(f'figures/orbit_lagrange_{nombre}.png', bbox_inches='tight', dpi = 300) # Formato vectorial preferido

        if animate:
            ext = 1.2
            lagrange = np.ones_like(sol1[:,0:3].T)
            lagrange[0,:], lagrange[1,:], lagrange[2,:] = lagrange[0,:] * x0, lagrange[1,:] * y0, lagrange[2,:]*0
            tierra = np.zeros_like(sol1[:,0:3].T)
            luna = np.ones_like(sol1[:,0:3].T)
            luna[0,:], luna[1,:], luna[2,:] = luna[0,:] * 1, luna[1,:] * 0, luna[2,:]*0
            
            ja.animar_trayectorias(
                datos=[sol1[:,0:3].T, sol2[:,0:3].T, lagrange, tierra, luna], 
                duracion=5.0, fps=30,
                archivo=f"figures/orbit_lagrange_{nombre}_anim.mp4",
                title=rf"Evolución alrededor de {nombre}",
                xlabel=r"$x$ [NU]", ylabel=r"$y$ [NU]", zlabel=r"$z$ [NU]",
                xlim = (x0 - 2, x0 + 0.5) if nombre != 'L4' else (x0 - 0.02, x0 + 0.02),
                ylim = (y0 - 1, y0 + 1) if nombre != 'L4' else (y0 - 0.02, y0 + 0.02),
                zlim = (-1,1),
                figsize = (15, 15),
                colors=['violet', 'lawngreen', 'black', 'green', 'gold'],  labels=[r'$\delta_1$', r'$\delta_2$', nombre, 'Tierra', 'Luna'],
                linewidth = 1,
                guardar = True if save_figs else False, dpi = 150
            )

        plt.show()
        plt.close()


# --- Pozo de Potencial Efectivo ---

if pozo_pot:
    pts_evaluar = {k: pts_lagrange[k] for k in ['L1', 'L2', 'L3', 'L4', 'L5']}
    fig2 = plt.figure(figsize=(8, 6))
    fig2.canvas.manager.set_window_title('Pozo de Potencial')

    fig_proj, ax_proj = plt.subplots(figsize=(8,6))
    
    ax_pot = fig2.add_subplot(111, projection='3d')

    X_mesh = np.linspace(-1.5, 1.5, 200)
    Y_mesh = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(X_mesh, Y_mesh)

    Z = U_eff(X, Y)
    Z = np.clip(Z, -3, np.max(Z)) 


    ax_pot.plot_surface(X, Y, Z, cmap='spring', edgecolor='none', alpha=0.5)
    ax_pot.contourf(X, Y, Z, zdir='z', offset=-3, cmap='coolwarm', alpha =0.5)

    ax_proj.contourf(X, Y, Z, cmap='coolwarm')

    for (nombre, (x0, y0)), marker, color in zip(pts_evaluar.items(), ['1', '2', '3', 6, 7], ['green', 'blue', 'magenta','k', 'k']):
        ax_pot.scatter3D(x0, y0, U_eff(x0,y0), marker = marker, s = 40, label = f'{nombre}', zorder=1, color = color)
        ax_proj.scatter(x0, y0, marker=marker, s=40, color=color, label=nombre) # Proyección en el contour


    ax_pot.set_title(r'Pozo de Potencial Efectivo $U_{eff}(x,y)$')
    ax_pot.set_xlabel(r'$x$ [NU]')
    ax_pot.set_ylabel(r'$y$ [NU]')
    ax_pot.set_zlabel(r'$U_{eff}$ [NU]')
    ax_pot.legend()

    ax_proj.plot(0,0, label = 'Tierra', marker = 'o', ms = 20, color = 'green')
    ax_proj.plot(1,0, label = 'Luna', marker = 'o', ms = 10, color = 'gold')
    ax_proj.set_xlabel(r'$x$ [NU]')
    ax_proj.set_ylabel(r'$y$ [NU]')
    ax_proj.set_title(r'Proyección 2D de $U_{eff}(x,y)$')
    ax_proj.legend()
    ax_proj.set_aspect('equal')

    if save_figs:
        fig2.savefig("figures/potencial_3d.png", dpi=300)
        fig_proj.savefig("figures/potencial_2d.png", dpi=300)

    plt.tight_layout()
    plt.show()


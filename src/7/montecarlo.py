# %% Imports y Declaraciones
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import linregress
import sys
from pathlib import Path
import importlib
from matplotlib.ticker import MaxNLocator

figuras = Path(__file__).resolve().parent.parent / '7' / 'figures'
figuras.mkdir(parents=True, exist_ok=True)  

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
import tools as ja
importlib.reload(ja)
ja.setup_style(base_size=19, dpi=120)

np.random.seed(19)

# === Parámetros libres del código ===
testing = 0
save_figs = 0


ball = 0

# ================================
# INTEGRACIÓN MEDIANTE MONTECARLO
# ================================

# Variables globales
N_samples = 5*10**4 if not testing else 10**1 # Número de puntos aleatorios para montecarlo bola
N_samples_func = 10**3 if not testing else 10**2 # Número de puntos aleatorios para montecarlo función
radius = 1
dimensions = np.arange(0,10)

# Funciones auxiliares
def inside_ball(coords, radius):
    '''
    Para un vector de N_samples partículas de dimensión d, evalúa su pertenencia a la hiperesfera

    input:
        - coords: ndarray (N_samples, d) 
        - radius: float
    
    output:
        - inside: boolean matrix
    '''
    d = len(coords.T) 

    r = np.sum(coords**2, axis =1)
    inside = r < radius

    return inside


def circle(radius):
    theta = np.linspace(0, 2* np.pi, 10**4) 
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    return x, y

def sphere(radius, res = 10**2):
    theta = np.linspace(0, np.pi, res)
    phi = np.linspace(0, 2*np.pi, res)

    THETA, PHI = np.meshgrid(theta, phi)

    x = radius * np.sin(THETA) * np.cos(PHI)
    y = radius * np.sin(THETA) * np.sin(PHI)
    z = radius * np.cos(THETA)

    return x, y, z


def integral_montecarlo(func, boundaries, N_samples):
    '''
    Función para realizar integrales definidas empleando el método de Montecarlo junto con su desviación estándar
    
    input:
        - func: función a integrar
        - boundaries: ndarray (N_dimensions, 2)
        - N_samples: número de muestreos para el método montecarlo

    output: 
        - resultado de la integral definida 
    '''
    boundaries = np.atleast_2d(boundaries)
    long = len(boundaries)
    

    vol = np.prod(boundaries[:, 1] - boundaries[:, 0])
    rand = np.random.uniform(boundaries[:,0], boundaries[:,1], (N_samples, long))
    eval = func(rand)
    I = vol * np.mean(eval, axis =0)
    std = vol * np.std(eval, axis = 0) / np.sqrt(N_samples)
    return I, std

# ===========================================

# Cálculo del hipervolumen de una bola en dimensión d
if ball:
    vol = np.zeros_like(dimensions, dtype='float')

    # Bucle en distintas dimensiones:
    for i, d in enumerate(dimensions):
        
        # Para N_samples, calculo la posición de x_1, x_2, ..., x_d, limitado a +- radius (hipercubo)
        coord = np.random.uniform(-radius, radius, (N_samples, d))


        inside = inside_ball(coord, radius) 
        frac = np.sum(inside)/N_samples
        hypervol = frac * (2.0*radius) ** float(d) # Fórmula del hipervolumen de dimensión d
        vol[i] = hypervol.astype(float)


        if d==2:
            # Caso particular d = 2
            inside_plot = coord[inside]
            x, y = circle(radius)

            fig, ax = plt.subplots(figsize=(10,10))
            ax.scatter(coord[:,0], coord[:,1], color ='red', label='Fuera de la bola')
            ax.scatter(inside_plot[:,0], inside_plot[:,1], color='green', label='Dentro de la bola')
            ax.plot(x, y, label= 'Frontera de la bola', color = 'cyan', lw=4)    
            ax.legend(loc='upper right')
            ax.set_xlabel(r'Posición $x$')
            ax.set_ylabel(r'Posición $y$')
            ax.set_aspect('equal')
            ax.set_title(rf'Pertenencia a bola de dimensión {d}')

            if save_figs:
                fig.savefig("figures/ball_2d.png", dpi=300, bbox_inches='tight')


        if d == 3:
            # Caso particular d = 3
            inside_plot = coord[inside]
            x, y, z = sphere(radius)

            fig = plt.figure(figsize=(10,10))
            ax1 = fig.add_subplot(projection = '3d')
            ax1.scatter(coord[:,0], coord[:,1], coord[:,2], color ='red', label='Fuera de la bola', alpha = 0.1)
            ax1.scatter(inside_plot[:,0], inside_plot[:,1], inside_plot[:,2], color='green', label='Dentro de la bola')
            ax1.plot(x, y, z, label = 'Frontera de la bola', color = 'cyan', lw=4, alpha = 0.4)    
            ax1.legend(loc='upper right')
            ax1.set_xlabel(r'Posición $x$', labelpad=25)
            ax1.set_ylabel(r'Posición $y$', labelpad=25)
            ax1.set_zlabel(r'Posición $z$', labelpad=25)
            ax1.set_aspect('equal')
            ax1.set_title(rf'Pertenencia a bola de dimensión {d}')
            
            ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax1.zaxis.set_major_locator(MaxNLocator(nbins=5))

            if save_figs:
                fig.savefig("figures/ball_3d.png", dpi=300, bbox_inches='tight')


    vmax = np.argmax(vol)

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.plot(dimensions, vol, '-.o', color='violet', lw = 3)
    ax2.plot(dimensions[vmax], vol[vmax], '*', color = 'gold', ms = 20, label='Mayor Hipervolumen')
    ax2.set_xlabel('Dimensión del espacio')
    ax2.set_ylabel('Hipervolumen') # En cada dimensión tendrá unas unidades distintas
    ax2.set_title('Hipervolumen en función de la dimensión')
    ax2.legend()
    if save_figs:
        fig.savefig("figures/ball_hypervolume.png", dpi=300, bbox_inches='tight')
    plt.show()



# Evaluación de una integral definida mediante montecarlo:
def x2(x):
    return x**2

I, std = integral_montecarlo(x2, [0,1], N_samples)

print(I, std)
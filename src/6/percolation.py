# %% Imports y Declaraciones
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy as sp
from scipy.stats import linregress
import sys
from pathlib import Path
import importlib
from matplotlib.ticker import MaxNLocator
import heapq

figuras = Path(__file__).resolve().parent.parent / '6' / 'figures'
figuras.mkdir(parents=True, exist_ok=True)  

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
import tools as ja
importlib.reload(ja)
ja.setup_style(base_size=19, dpi=120)

#================================================

save_figs = 1

# Variables y Funciones

L = 200 # Tamaño del grid        
steps = 5000

def invasion_percolation(L, steps, neighborhood='vn', distribution=np.random.rand ,seed_val=None):
    if seed_val:
        np.random.seed(seed_val)

    R = distribution(L, L) # Campo de resistencias con ruido   
    
    # Boolean Matrix
    cluster = np.zeros((L, L), dtype=bool)
    x0, y0 = L // 2, L // 2
    cluster[x0, y0] = True
     
    boundary = [] # Cola de prioridad para la frontera (resistencia, x, y)

    if neighborhood == 'vn': # Vecindad de Von Neumann (distancia 1, sin diagonales)
        neighbors = [(-1,0), (1,0), (0,-1), (0,1)]
        def add_neighbors(x, y):
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                # Condición pertenencia al espacio + no invadido todavía
                if 0 <= nx < L and 0 <= ny < L and not cluster[nx, ny]:
                    # Guardanmos en Boundary el cuadrado (dentro de la vecindad) que menor resistencia ofrece
                    heapq.heappush(boundary, (R[nx, ny], nx, ny))

    elif neighborhood == 'moore': #Vecindad de Moore (distancia 1, con diagonales)
        raise NotImplementedError(f"To be implemented")
    else:
        raise NotImplementedError(f"No se ha implementado ese método de vecindad")
                    
    add_neighbors(x0, y0)
    
    # Evolución temporal
    invasion_sequence = [(x0, y0)]
    
    for i in range(steps):
        if not boundary:
            break # El fluido ha colapsado o llenado el espacio
            
        # Extraemos nodo de mínima resistencia
        r, x, y = heapq.heappop(boundary)
        
        if not cluster[x, y]:
            cluster[x, y] = True
            invasion_sequence.append((x, y))
            add_neighbors(x, y)
            
    return cluster, invasion_sequence, R

  
cluster, seq, R = invasion_percolation(L, steps, seed_val = 19)


#================================================

import numpy.ma as ma
plt.figure(figsize=(8, 8))
plt.imshow(R, cmap = 'inferno', alpha=0.2)
fluido_invasor = ma.masked_where(~cluster, cluster)
plt.imshow(fluido_invasor, cmap='Blues', interpolation='none', vmin=0, vmax=1)

if save_figs:
    plt.savefig("figures/percolation", dpi =300, bbox_inches='tight')
plt.title(f'Percolación de Invasión: Interacción de Fluidos ($N = {steps}$)')
plt.axis('off')
plt.tight_layout()
plt.show()

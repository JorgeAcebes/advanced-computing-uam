# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

# Parámetros del sistema espacial
L = 50  # Longitud característica del grid L x L

# Decidir qué tipo de inicio:
random_start = 0

if random_start:
    # np.random.seed(19)
    p_activa = 0.2  # Probabilidad de estado activo inicial
    grid = np.random.choice([0, 1], size=(L, L), p=[1 - p_activa, p_activa])
else:
    gc = int(L/2) # Centro del grid
    grid = np.zeros([L,L])
    grid[gc-1:gc+2, gc-1:gc+2] = np.array([[0, 1, 0],
                                      [0, 0, 1], 
                                      [1, 1, 1]])

# Matriz de convolución para vecindad de Moore
kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

def evolucion_temporal(frame):
    """Calcula el estado del sistema en t+1 dadas las condiciones en t."""
    global grid
    
    # Campo escalar de vecinos activos N_ij(t) mediante convolución
    vecinos = convolve2d(grid, kernel, mode='same', boundary='wrap')
    
    # Generación del nuevo microestado s(t+1) aplicando las reglas locales
    nueva_grid = np.zeros_like(grid)
    
    # Regla de supervivencia y reproducción (evaluación booleana vectorizada)
    supervivencia = (grid == 1) & ((vecinos == 2) | (vecinos == 3))
    reproduccion = (grid == 0) & (vecinos == 3)
    
    nueva_grid[supervivencia | reproduccion] = 1 
    
    grid = nueva_grid
    img.set_data(grid)
    return img,
# Visualización del sistema espacial
fig, ax = plt.subplots(figsize=(6, 6))

# Discretización del plano en L+1 fronteras semienteras
fronteras = np.arange(-0.5, L, 1)
ax.set_xticks(fronteras)
ax.set_yticks(fronteras)

# Supresión de etiquetas preservando el espacio métrico
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis='both', length=0)

# Mapeo del microestado y proyección de la cuadrícula
# Se fuerza zorder=0 para la matriz y zorder=1 para la rejilla
img = ax.imshow(grid, cmap='binary', interpolation='nearest', zorder=0)
ax.grid(color='gray', linestyle='-', linewidth=0.5, zorder=1)

# Integración temporal
# Al inhabilitar la optimización (blit=False) garantizamos la invariancia de la rejilla
ani = animation.FuncAnimation(fig, evolucion_temporal, interval=50, blit=False, cache_frame_data=False)
plt.show()
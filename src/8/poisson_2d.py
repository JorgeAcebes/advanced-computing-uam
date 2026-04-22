import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))

from tools import solve_poisson_sor
from tools import setup_style

setup_style()

def plot_potential(V, title, extent=None):
    plt.imshow(V, cmap='coolwarm', origin='upper', extent=extent)
    plt.colorbar(label='Potencial (V)')
    plt.title(title)
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.show()

# =====================================================================
# Definición Topológica: Ejercicio 9.1 (Poisson)
# =====================================================================
def run_exercise_9_1():
    N = 101  # Red 1m x 1m con a=1cm (100 intervalos -> 101 nodos)
    a = 0.01 

    L = N - 1
    extent = [0, L, 0, L]
    
    V = np.zeros((N, N)) # Creamos una matriz de ceros
    rho = np.zeros((N, N)) # La densidad también es una matriz de ceros.
    is_boundary = np.zeros((N, N), dtype=bool) # Inicializamos nuestra matriz de boundary como tipo bool e inicialmente todo False

    # Fronteras a 0V
    is_boundary[0, :] = is_boundary[-1, :] = is_boundary[:, 0] = is_boundary[:, -1] = True # Establecemos que la 
    # primera/última fila y columna sean true
    
    # Distribución de densidad de carga
    rho[20:41, 60:81] = 1.0   # Carga Positiva
    rho[60:81, 20:41] = -1.0  # Carga Negativa


    V = solve_poisson_sor(V, rho, is_boundary, a, epsilon0=1.0, tol=1e-6, omega=1.9)
    plot_potential(V, "Ej 9.1: Ecuación de Poisson", extent)

# =====================================================================
# Definición Topológica: Ejercicio 9.2 (Laplace)
# =====================================================================
def run_exercise_9_2():
    N = 101  
    a = 0.01

    L = N - 1
    extent = [0, L, 0, L]
    
    V = np.zeros((N, N))
    rho = np.zeros((N, N))  # Laplace implica rho = 0
    is_boundary = np.zeros((N, N), dtype=bool)

    # Definición de fronteras físicas
    is_boundary[0, :] = is_boundary[-1, :] = is_boundary[:, 0] = is_boundary[:, -1] = True
    
    # Condiciones de Dirichlet
    V[0, :] = 1.0  # Pared superior a 1V

    V = solve_poisson_sor(V, rho, is_boundary, a, epsilon0=1.0, tol=1e-6, omega=1.9)
    plot_potential(V, "Ej 9.2: Ecuación de Laplace", extent)

if __name__ == "__main__":
    run_exercise_9_1()
    run_exercise_9_2()
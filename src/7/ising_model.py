# %% Imports y Declaraciones
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
from pathlib import Path
import importlib
from numba import njit
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator

figuras = Path(__file__).resolve().parent.parent / '7' / 'figures'
figuras.mkdir(parents=True, exist_ok=True)  

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
import tools as ja
importlib.reload(ja)
ja.setup_style(base_size=19, dpi=120)

np.random.seed(19)


# %%
# Parámetros globales
J = 1.0
T_c_teo = 2.0 / np.log(1.0 + np.sqrt(2.0)) # ~2.269 (Temperatura de Courie teórica)
L_list = [5, 10, 20, 30, 40] # Posibles longitudes del lado de la matriz de espines
eq_steps = 2000   # Pasos para termalizar
mc_steps = 10000  # Pasos de medida

# Rango de temperaturas (más fino alrededor de T_c)
T_rango = np.concatenate((
    np.linspace(1.5, 2.1, 15),
    np.linspace(2.15, 2.4, 30), # Fino cerca de T_c
    np.linspace(2.45, 3.2, 15)
))

@njit
def paso_metropolis(red, T):
    """Realiza un barrido completo (N intentos) sobre la red."""
    L = red.shape[0]
    for _ in range(L * L):
        i = np.random.randint(L)
        j = np.random.randint(L)
        s = red[i, j]
        
        # Condiciones de contorno periódicas
        vecinos = red[(i+1)%L, j] + red[i, (j+1)%L] + red[(i-1)%L, j] + red[i, (j-1)%L]
        dE = 2.0 * J * s * vecinos
        
        # Criterio de aceptación
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            red[i, j] = -s
    return red

@njit
def calcular_energia(red):
    """Calcula la energía total de la configuración actual."""
    L = red.shape[0]
    E = 0.0
    for i in range(L):
        for j in range(L):
            S = red[i, j]
            vecinos = red[(i+1)%L, j] + red[i, (j+1)%L]
            E += -J * S * vecinos
    return E

def simular_ising(L, T_array):
    """Simula la red para un L dado y un array de temperaturas."""
    N = L * L
    C_N = np.zeros(len(T_array))
    
    for idx, T in enumerate(T_array):
        # Inicialización aleatoria (alta temperatura)
        red = np.random.choice(np.array([-1, 1]), size=(L, L)) # Escogemos +- 1 para cada espín
        
        # Termalización
        for _ in range(eq_steps):
            red = paso_metropolis(red, T)
            
        # Medida
        E_sum = 0.0
        E2_sum = 0.0
        for _ in range(mc_steps):
            red = paso_metropolis(red, T)
            E = calcular_energia(red)
            E_sum += E
            E2_sum += E**2
            
        E_prom = E_sum / mc_steps
        E2_prom = E2_sum / mc_steps
        
        # C/N = ( <E^2> - <E>^2 ) / (N * T^2)
        C_N[idx] = (E2_prom - E_prom**2) / (N * T**2)
        
    return C_N

# Función para el ajuste lineal
def ajuste_lineal(x, a, b):
    return a * x + b

# Ejecución y análisis
C_max_list = []

plt.figure(figsize=(12, 5))

# --- Gráfica 1: C/N vs T ---
plt.subplot(1, 3, 1)
for L in L_list:
    C_N_T = simular_ising(L, T_rango)
    plt.plot(T_rango, C_N_T, marker='o', markersize=4, label=f'L={L}')
    
    # Extraer el C_max para la segunda parte
    C_max_list.append(np.max(C_N_T))

plt.axvline(T_c_teo, color='k', linestyle='--', label=r'$T_c$ teórica')
plt.xlabel('Temperatura ($T$)')
plt.ylabel(r'$C/N$')
plt.title('Capacidad Calorífica vs Temperatura')
plt.legend()
plt.grid(True, alpha=0.5)

# --- Gráfica 2: C_max/N vs log(L) ---
plt.subplot(1, 3, 2)
log_L = np.log(L_list)
C_max_array = np.array(C_max_list)

# Ajuste por mínimos cuadrados
popt, pcov = curve_fit(ajuste_lineal, log_L, C_max_array)
a, b = popt

sign = '+' if b >= 0 else '-'
b = np.abs(b)
plt.plot(log_L, C_max_array, 'ko', label='Datos simulados')
plt.plot(log_L, ajuste_lineal(log_L, a, b), 'r-', label=f'Ajuste: $y={a:.3f}x{sign}{b:.3f}$')

plt.xlabel(r'$\log L$')
plt.ylabel(r'$C_{\text{máx}}/N$')
plt.title(r'Divergencia logarítmica de $C_{\text{máx}}/N$')
plt.legend()
plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()



# if save_figs:
#     fig.savefig("figures/multiplot_analysis_rw.png", dpi=300, bbox_inches='tight')
#     figcmap.savefig("figures/diffusion_cmap_rw.png", dpi=300, bbox_inches='tight')

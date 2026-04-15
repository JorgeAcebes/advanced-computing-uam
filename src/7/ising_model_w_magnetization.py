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


save_figs = 0

# %%
# Parámetros globales
J = 1.0
T_c_teo = 2.0 / np.log(1.0 + np.sqrt(2.0)) # ~2.269 (Temperatura de Courie teórica)
L_list = [4, 7] # Posibles longitudes del lado de la matriz de espines. Lo ponemos como lista para poder ampliarlo
eq_steps = 2000   # Pasos para termalizar
mc_steps = 10000  # Pasos de medida

# Rango de temperaturas (más fino alrededor de T_c)
T_rango = np.concatenate((
    np.linspace(0.5, 2.1, 35),
    np.linspace(2.15, 2.4, 30), # Fino cerca de T_c
    np.linspace(2.45, 10.2, 150)
))

@njit
def paso_metropolis(red, T):
    """Realiza un barrido completo (N=L*L intentos) sobre la red."""
    L = red.shape[0]
    for _ in range(L * L):
        # Escogemos una celda al azar
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
def paso_metropolis_sistemático(red, T):
    """Realiza un barrido completo (N=L*L intentos) sistemático (permitimos que todos los 
    espines tengan la posibilidad de invertirse) sobre la red."""
    L = red.shape[0]
    for i in range(L):
        for j in range(L):
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


@njit
def calcular_magnetization(red):
    """Calcula la magnetización total de la configuración actual."""
    L = red.shape[0]
    M = 0.0
    for i in range(L):
        for j in range(L):
            S = red[i, j]
            M += S
    return M


def simular_ising(L, T_array):
    """Simula la red para un L dado y un array de temperaturas."""
    N = L * L
    C_N = np.zeros(len(T_array))
    chi = np.zeros(len(T_array))
    E_vec = np.zeros(len(T_array))
    M_vec = np.zeros(len(T_array))
    
    for idx, T in enumerate(T_array):
        # Inicialización aleatoria (alta temperatura)
        red = np.random.choice(np.array([-1, 1]), size=(L, L)) # Escogemos +- 1 para cada espín
        
        # Termalización
        for _ in range(eq_steps):
            red = paso_metropolis(red, T)
            
        # Medida
        E_sum = 0.0
        E2_sum = 0.0

        M_sum = 0.0
        M2_sum = 0.0

        for _ in range(mc_steps):
            red = paso_metropolis(red, T)
            E = calcular_energia(red)
            M = calcular_magnetization(red)
            E_sum += E
            E2_sum += E**2
            M_sum += np.abs(M)
            M2_sum += M**2
            
        E_prom = E_sum / mc_steps
        E2_prom = E2_sum / mc_steps

        M_prom = M_sum / mc_steps
        M2_prom = M2_sum / mc_steps
        
        E_vec[idx] = E_prom / N
        M_vec[idx] = M_prom / N

        # C/N = ( <E^2> - <E>^2 ) / (N * T^2)
        C_N[idx] = (E2_prom - E_prom**2) / (N * T**2)
        
        # chi/N = ( <M^2> - <M>^2 ) / (N * T)        
        chi[idx] = (M2_prom - M_prom**2) / (N * T)
    return C_N, chi, E_vec, M_vec

# Función para el ajuste lineal
def ajuste_lineal(x, a, b):
    return a * x + b

# Ejecución y análisis
C_max_list = []
chi_max_list = []


plt.figure(figsize=(18, 10))

# Definimos los subplots de antemano para mayor claridad
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)
ax3 = plt.subplot(2, 3, 3)
ax4 = plt.subplot(2, 3, 4)
ax5 = plt.subplot(2, 3, 5)
ax6 = plt.subplot(2, 3, 6)


for L in L_list:

    C_N_T, chi_T, E_vec, M_vec = simular_ising(L, T_rango)
    
    # Guardamos el máximo para la gráfica 3 y 6
    C_max_list.append(np.max(C_N_T))
    chi_max_list.append(np.max(chi_T))

    ax1.plot(T_rango, E_vec, marker='o', markersize=3, label=f'L={L}')
    ax2.plot(T_rango, C_N_T, marker='o', markersize=3, label=f'L={L}')
    ax4.plot(T_rango, M_vec, marker='o', markersize=3, label=f'L={L}')
    ax5.plot(T_rango, chi_T, marker='o', markersize=3, label=f'L={L}')


# --- Formateo Gráfica 1: Energía ---
ax1.axvline(T_c_teo, color='k', linestyle='--', label=r'$T_c$ teórica')
ax1.set_xlabel('Temperatura ($T$)')
ax1.set_ylabel(r'Energía')
ax1.set_title(r'Energía en función de $T$')
ax1.legend()
ax1.grid(True, alpha=0.3)



# --- Formateo Gráfica 2: C/N ---
ax2.axvline(T_c_teo, color='k', linestyle='--', label=r'$T_c$ teórica')
ax2.set_xlabel('Temperatura ($T$)')
ax2.set_ylabel(r'$C/N$')
ax2.set_title('Capacidad Calorífica')
ax2.legend()
ax2.grid(True, alpha=0.3)


# --- Gráfica 3: Escala de Tamaño Finito (FSS) ---
log_L = np.log(L_list)
C_max_array = np.array(C_max_list)

# Ajuste lineal para verificar la divergencia logarítmica
popt, _ = curve_fit(ajuste_lineal, log_L, C_max_array)
a, b = popt

ax3.plot(log_L, C_max_array, 'ko', label='Datos simulados')
ax3.plot(log_L, ajuste_lineal(log_L, a, b), 'r-', 
         label=f'Ajuste: $y={a:.3f}x {["-","+"][b>0]} {abs(b):.3f}$')

ax3.set_xlabel(r'$\log L$')
ax3.set_ylabel(r'$C_{\text{máx}}/N$')
ax3.set_title(r'Escalado de $C_{\text{máx}}/N$')
ax3.legend()
ax3.grid(True, alpha=0.3)


# --- Formateo Gráfica 4: Magnetización ---
ax4.axvline(T_c_teo, color='k', linestyle='--', label=r'$T_c$ teórica')
ax4.set_xlabel('Temperatura ($T$)')
ax4.set_ylabel('Magnetización')
ax4.set_title(r'Magnetización en función de $T$')
ax4.legend()
ax4.grid(True, alpha=0.3)


# --- Formateo Gráfica 5: chi/N ---
ax5.axvline(T_c_teo, color='k', linestyle='--', label=r'$T_c$ teórica')
ax5.set_xlabel('Temperatura ($T$)')
ax5.set_ylabel(r'$\chi/N$')
ax5.set_title('Susceptibilidad Magnética')
ax2.legend()
ax5.grid(True, alpha=0.3)


# --- Gráfica 3: Escala de Tamaño Finito (FSS) ---
log_L = np.log(L_list)
chi_max_array = np.array(chi_max_list)

# Ajuste lineal para verificar la divergencia logarítmica
popt_mag, _ = curve_fit(ajuste_lineal, log_L, chi_max_array)
c, d = popt_mag

ax6.plot(log_L, chi_max_array, 'ko', label='Datos simulados')
ax6.plot(log_L, ajuste_lineal(log_L, c, d), 'r-', 
         label=f'Ajuste: $y={c:.3f}x {["-","+"][d>0]} {abs(d):.3f}$')

ax6.set_xlabel(r'$\log L$')
ax6.set_ylabel(r'$\chi_{\text{máx}}/N$')
ax6.set_title(r'Escalado de $\chi_{\text{máx}}/N$')
ax6.legend()
ax6.grid(True, alpha=0.3)


if save_figs:
    plt.savefig("figures/multiplot_analysis_rw.png", bbox_inches='tight')

plt.tight_layout()
plt.show()



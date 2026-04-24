# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from numba import njit
from matplotlib.ticker import MaxNLocator

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))

from tools import gauss_seidel_cilindrico
from tools import setup_style

setup_style()

# =====================================================
# Ruta hacia la carpeta de data
current_dir = Path(__file__).resolve().parent

# Ruta de figuras
figuras = Path(__file__).resolve().parent.parent / '8' / 'figures'
figuras.mkdir(parents=True, exist_ok=True)  
# =====================================================


# Parámetros físicos
Ri, Re, L = 1.0, 4.0, 20.0 # cm
Vi, Ve = 2.5, 0.0 # V (potencial in, potencial out)
Vb, Vt = 0.0, 0.0 # V (potencial bottom, potencial top)

# Parámetros de la malla
Nr, Nz = 100, 200 # Distintos puntos para la dirección radial y axial
dr = (Re - Ri) / Nr # Infinitesimal en dirección radial
dz = L / Nz # Infinitesimal en dirección axial

r = np.linspace(Ri, Re, Nr+1)
z = np.linspace(0, L, Nz+1)

# Calculamos el potencial en todo el dominio
V_num = gauss_seidel_cilindrico(Nr, Nz, r, dr, dz, Vi=Vi, Ve=Ve, Vb=Vb, Vt=Vt)

# Extraemos el perfil radial en el centro del cilindro (z = L/2)
V_z_mid = V_num[:, Nz//2]

# Calculamos la aproximación analítica (cilindro infinito --> tapas no influyen) para comparar 
V_analitico = Vi * np.log(Re / r) / np.log(Re / Ri)


# === GRÁFICAS ===

# --- V(r, z = L/2) ---
plt.figure(figsize=(8, 5))
plt.plot(r, V_z_mid, 'k-', linewidth=2, label='Numérico ($z = L/2$)')
plt.plot(r, V_analitico, 'r--', linewidth=2, label='Analítico ($L \\to \\infty$)')

plt.title('Potencial Electrostático Radial en Condensador Cilíndrico')
plt.xlabel('$r$ [cm]')
plt.ylabel('$V(r, z = L/2)$ [V]')
plt.xlim(Ri, Re)
plt.ylim(0, Vi)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig(figuras/'pot_radial_z_cte.png')
plt.show()


# --- Distribución Potencial ---
# Toda la información que necesitamos debido a la simetría en theta
plt.figure(figsize=(8, 6))
img = plt.imshow(V_num.T, 
                extent=[Ri, Re, 0, L],
                 origin='lower', 
                 cmap='inferno', 
                 aspect='auto',
                 interpolation='nearest')

plt.colorbar(img, label=r'Potencial $V(r, z)$')
plt.xlabel('$r$ [cm]')
plt.ylabel('$z$ [cm]')
plt.title(r'Distribución del Potencical Electrostático $V$')
plt.savefig(figuras/'pot_elec_2d.png')
plt.show()


# Extendemos la solución para 3D:
N_theta = 60
theta = np.linspace(0, 2*np.pi, N_theta)

# Mallas tridimensionales
R_grid, Theta_grid, Z_grid = np.meshgrid(r, theta, z, indexing='ij')
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)

# Propagamos la solución 2D a lo largo del ángulo theta (axis=1)
V_3D = np.repeat(V_num[:, np.newaxis, :], N_theta, axis=1)

# Sección longitudinal
# Mantenemos solo los puntos donde Y >= 0
mask = Y >= 0
step = 3 # Paso de 3 para que no pete mi ordenador graficando 600.000 puntos

X_plot = X[mask].flatten()[::step]
Y_plot = Y[mask].flatten()[::step]
Z_plot = Z_grid[mask].flatten()[::step]
V_plot = V_3D[mask].flatten()[::step]

# --- Gráfica 3d ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
sc = ax.scatter(X_plot, Y_plot, Z_plot, c=V_plot, cmap='magma', s=2, alpha=0.6)

# Configuraciones de los ejes
ax.set_xlabel('$x$ [cm]', labelpad=10)
ax.set_ylabel('$y$ [cm]', labelpad=5)
ax.set_zlabel('$z$ [cm]', labelpad=35)
ax.set_title(r'Potencial Electrostático $V(r, \theta, z)$ (Sección Longitudinal)', pad=30)

ax.tick_params(axis='y', pad=0)
ax.tick_params(axis='z', pad=15)

ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=2))

# Ajuste de escala
ax.set_xlim(-Re, Re)
ax.set_ylim(0, Re) 
ax.set_zlim(0, L)
ax.set_box_aspect([2*Re, 1.5*Re, L]) 

# Ajustes de la Colorbar
cbar = plt.colorbar(sc, ax=ax, pad=0.0, shrink=0.75) 
cbar.set_label(r'$V(r,\theta,z)$ [V]', labelpad=20)

plt.savefig(figuras/'pot_elec_3d.png')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from numba import njit


path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))

from tools import gauss_seidel_cilindrico
from tools import setup_style

setup_style()

# Parámetros físicos
Ri, Re, L = 1.0, 4.0, 20.0  # cm
Vi, Ve = 2.5, 0.0           # V (potencial in, potencial out)
Vb, Vt = 0.0, 0.0           # V (potencial bottom, potencial top)

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

# Calculamos la aproximación analítica (cilindro infinito) para comparar
V_analitico = Vi * np.log(Re / r) / np.log(Re / Ri)

plt.figure(figsize=(8, 5))
plt.plot(r, V_z_mid, 'k-', linewidth=2, label='Numérico ($z = L/2$)')
plt.plot(r, V_analitico, 'r--', linewidth=2, label='Analítico ($L \\to \\infty$)')

plt.title('Potencial electrostático radial en el condensador cilíndrico')
plt.xlabel('$r$ (cm)')
plt.ylabel('$V(r)$ (V)')
plt.xlim(Ri, Re)
plt.ylim(0, Vi)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()

# 3. Mapeo a tensor 3D (r, theta, z) -> (x, y, z)
N_theta = 60
theta = np.linspace(0, 2*np.pi, N_theta)

# Mallas tridimensionales
R_grid, Theta_grid, Z_grid = np.meshgrid(r, theta, z, indexing='ij')
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)

# Propagar la solución 2D a lo largo del ángulo theta (axis=1)
V_3D = np.repeat(V_num[:, np.newaxis, :], N_theta, axis=1)

# 4. Filtrado espacial (Corte longitudinal para ver el interior)
# Mantenemos solo los puntos donde Y >= 0
mask = Y >= 0
step = 3 

X_plot = X[mask].flatten()[::step]
Y_plot = Y[mask].flatten()[::step]
Z_plot = Z_grid[mask].flatten()[::step]
V_plot = V_3D[mask].flatten()[::step]


# 5. Visualización 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot mapeando el potencial al mapa de colores
sc = ax.scatter(X_plot, Y_plot, Z_plot, c=V_plot, cmap='magma', s=2, alpha=0.6)

# Configuraciones físicas de los ejes
ax.set_xlabel('$X$ (cm)')
ax.set_ylabel('$Y$ (cm)')
ax.set_zlabel('$Z$ (cm)')
ax.set_title('Potencial Electrostático $V(x,y,z)$ (Corte Interior)')

# Ajustar escala para evitar deformación del cilindro
ax.set_xlim(-Re, Re)
ax.set_ylim(0, Re) # Límite ajustado por el corte
ax.set_zlim(0, L)
ax.set_box_aspect([2*Re, Re, L]) 

cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
cbar.set_label('$V(x,y,z)$ (Voltios)')

plt.tight_layout()
plt.show()
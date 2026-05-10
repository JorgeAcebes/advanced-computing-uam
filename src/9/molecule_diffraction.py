import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================
# CONFIGURACIÓN GRÁFICA (Matplotlib)
# ==========================================
base_size = 14
# Desactivamos usetex para evitar cuelgues externos. Mathtext se encargará de las fórmulas.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.formatter.use_mathtext": True,
    "figure.dpi": 300,
    "xtick.labelsize": base_size,
    "ytick.labelsize": base_size,
    "axes.labelsize": base_size + 1,
    "axes.titlesize": base_size + 2,
    "legend.fontsize": base_size - 1,
    "figure.titlesize": base_size + 4
})

# Variable para controlar si se guardan los archivos (0: No, 1: Sí)
save = 1

# ==========================================
# PARÁMETROS FÍSICOS: NANOGRAFENO (HBC)
# ==========================================
h_bar = 1.0 
# Masa del Carbono * 42 átomos (Aprox. molécula HBC aislada)
mass = 12.0 * 42  
v_most_probable = 0.5 
v_spread = 0.01   # Alta coherencia longitudinal necesaria

grating_period = 2.5  # d (Periodo de la rejilla)
slit_width = 0.05     # a (Ancho de rendija)
N_slits = 40          # Número de rendijas
L_screen = 30.0       # Distancia a la pantalla

# Configuración de la pantalla
x_screen = np.linspace(-15, 15, 1200)
theta = np.arctan(x_screen / L_screen)
dx = x_screen[1] - x_screen[0]
bins_edges = np.append(x_screen, x_screen[-1] + dx)

# --- Cálculo de Intensidad (Lineal) ---
def diffraction_intensity(theta, lam, d, a, N):
    # Evitar división por cero en theta=0
    safe_theta = np.where(np.abs(theta) < 1e-12, 1e-12, theta)
    sin_theta = np.sin(safe_theta)
    
    beta = (np.pi * a / lam) * sin_theta
    delta = (2 * np.pi * d / lam) * sin_theta
    
    # Factor de forma (rendija) * Factor de estructura (interferencia)
    # Usamos np.sinc(x) que es sin(pi*x)/(pi*x) para el factor de forma
    # O la fórmula estándar cuidando el límite en beta=0.
    
    # Factor de forma: (sin(beta)/beta)^2
    form_factor = (np.sin(beta) / beta)**2
    
    # Factor de estructura: (sin(N*delta/2)/sin(delta/2))^2
    # Cuidar divisiones por cero cuando sin(delta/2) -> 0 (picos principales)
    denom_structure = np.sin(delta / 2)
    numer_structure = np.sin(N * delta / 2)
    
    # Manejo seguro de la división para el factor de estructura
    with np.errstate(invalid='ignore', divide='ignore'):
        structure_factor = np.where(np.abs(denom_structure) < 1e-8, N**2, (numer_structure / denom_structure)**2)
    
    return form_factor * structure_factor

# --- Espectro de velocidades (Policromaticidad) ---
velocities = np.linspace(v_most_probable - 3*v_spread, v_most_probable + 3*v_spread, 40)
weights = np.exp(-0.5 * ((velocities - v_most_probable) / v_spread)**2)
weights /= np.sum(weights) # Normalizar pesos

# Calcular intensidad total promediada en el espectro de velocidades
total_intensity = np.zeros_like(x_screen)
print("Calculando patrón teóroco policromático...")
for v, w in zip(velocities, weights):
    # Longitud de onda de de Broglie: lambda = h_bar * 2*pi / p = 2*pi / (m*v) (en unidades hbar=1)
    # O más estándar si hbar es h/2pi, lambda = h/p. Asumiremos h=2pi, hbar=1 => lambda = 2pi/(mv)
    lam = 2 * np.pi * h_bar / (mass * v)
    total_intensity += w * diffraction_intensity(theta, lam, grating_period, slit_width, N_slits)

# Normalizar intensidad máxima a 1
total_intensity /= np.max(total_intensity)
print("Cálculo completado.")


# ==========================================
# 1. ANIMACIÓN (EJE LINEAL)
# ==========================================
print("\nPreparando Animación (Figura 1)...")
# Usamos blit=False si da problemas, pero True es más rápido.
# Con usetex=False debería funcionar bien.
fig2, (ax_anim, ax_hist) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={'height_ratios': [1.5, 1]})

# Configuración área de vuelo de partículas
ax_anim.set_xlim(-15, 15)
ax_anim.set_ylim(-5, L_screen + 2)
ax_anim.axhline(0, color='gray', lw=3, linestyle='--', label='Rejilla difractora')
ax_anim.axhline(L_screen, color='black', lw=2, label='Pantalla detectora')
ax_anim.set_xlabel(r'Posición transversal $x$')
ax_anim.set_ylabel(r'Distancia de propagación $z$')
ax_anim.set_title('Simulación de Interferencia Cuántica de Nanografeno')
scat = ax_anim.scatter([], [], s=4, c='lime', alpha=0.6)
ax_anim.legend(loc='upper right')

# Configuración Histograma/Intensidad
ax_hist.set_xlim(-15, 15)
ax_hist.set_ylim(0, 1.1)
ax_hist.set_ylabel(r"Intensidad Normalizada $I/I_0$")
ax_hist.set_xlabel(r'Posición transversal $x$ sobre la pantalla')
hist_line, = ax_hist.plot(x_screen, np.zeros_like(x_screen), color='lime', lw=1.5, label='Detección Acumulada')
ax_hist.fill_between(x_screen, 0, total_intensity, color='cyan', alpha=0.1)
ax_hist.plot(x_screen, total_intensity, color='cyan', alpha=0.4, lw=0.8, linestyle='--', label='Perfil Teórico promediado')
ax_hist.legend(loc='upper right')

# Parámetros de la simulación de partículas
n_particles_per_frame = 50 # Reducido para que la animación sea más fluida en el renderizado
screen_hits = []

def update(frame):
    # Reiniciar la onda cada 60 frames
    current_wave = frame % 60
    # Posición z media del frente de onda en vuelo
    z_pos_mean = -5 + (L_screen + 5) * (current_wave / 55.0)
    
    # Muestreo de la distribución de probabilidad (basado en la intensidad teórica)
    prob_dist = total_intensity / np.sum(total_intensity)
    batch_x = np.random.choice(x_screen, size=n_particles_per_frame, p=prob_dist)
    
    if z_pos_mean < L_screen:
        # Partículas en vuelo: expansión cónica simple desde la rejilla (z=0)
        # Factor de expansión basado en z (burda aproximación visual)
        expansion_factor = (z_pos_mean + 5) / (L_screen + 5)
        x_pos = batch_x * expansion_factor
        # Añadir algo de dispersión en z para el "paquete"
        z_pos = np.random.normal(z_pos_mean, 0.5, size=n_particles_per_frame)
        scat.set_offsets(np.c_[x_pos, z_pos])
    else:
        # Partículas golpean la pantalla
        screen_hits.extend(batch_x)
        if len(screen_hits) > 0 and frame % 2 == 0: # Actualizar histograma cada 2 frames para rendimiento
            hist, _ = np.histogram(screen_hits, bins=bins_edges)
            max_h = np.max(hist)
            hist_line.set_ydata(hist / max_h if max_h > 0 else hist)
        # Limpiar scatter de partículas en vuelo
        scat.set_offsets(np.empty((0, 2)))
        
    return scat, hist_line

# Generar la animación
# Intervalo bajo (25ms) para buena velocidad visual. Reducir frames si tarda mucho.
ani = FuncAnimation(fig2, update, frames=300, blit=True, interval=30)

if save: 
    try:
        ani.save('figures/animation_nanographene.mp4', dpi=150, writer='ffmpeg')
        print("Guardado exitoso.")
    except Exception as e:
        print(f"Error al guardar: {e}. Asegúrate de tener ffmpeg instalado.")

plt.show()


# ============================================================
# 2. REPRESENTACIÓN ESTRUCTURA GRAFÉNICA (HBC) FUSIONADA
# ============================================================
print("\nGenerando Estructura Atómica Fusionada (Figura 2)...")
fig3 = plt.figure(figsize=(8, 7.25))
ax3 = fig3.add_subplot(111)

r_bond = 1.42 # Distancia C-C en grafeno (Angstroms)

# Definir la geometría básica de un hexágono con radio C-C
def get_hex_vertices(center, r_bond):
    # Orientación 'armchair' (vértice arriba/abajo) para que encajen al desplazarlos en X
    t = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 7) 
    return center[0] + r_bond*np.cos(t), center[1] + r_bond*np.sin(t)

# 1. Calcular centros de los 7 hexágonos (celda central + 6 periféricos)
dist_centers = np.sqrt(3) * r_bond # Distancia entre centros de hexágonos adyacentes
centers = [[0, 0]]
for i in range(6):
    angle = i * np.pi / 3 # Ángulos cada 60 grados
    centers.append([dist_centers*np.cos(angle), dist_centers*np.sin(angle)])

# 2. Recolectar todos los vértices (átomos) y redondear para evitar duplicados numéricos
all_vertices = []
for c in centers:
    hx, hy = get_hex_vertices(c, r_bond)
    # Redondeamos a 3 decimales para asegurar que los átomos compartidos tengan la misma coordenada
    for x, y in zip(hx[:-1], hy[:-1]): # Excluimos el último punto que repite el primero
        all_vertices.append((round(x, 3), round(y, 3)))

# Eliminar duplicados usando un conjunto (set) para obtener átomos únicos
unique_atoms = list(set(all_vertices))
unique_atoms_np = np.array(unique_atoms)

# 3. Dibujar todos los enlaces (bonds) recorriendo los hexágonos
# Al usar el radio r_bond completo, los hexágonos adyacentes compartirán líneas
for c in centers:
    hx, hy = get_hex_vertices(c, r_bond)
    # Dibujamos las líneas verdes de los hexágonos (enlaces C-C)
    ax3.plot(hx, hy, 'g-', lw=2.5, zorder=5) 

# 4. Dibujar los átomos únicos (carbonos) sobre los enlaces
ax3.scatter(unique_atoms_np[:, 0], unique_atoms_np[:, 1], color='black', s=50, zorder=10, label='Átomo de Carbono')

ax3.set_title(r"Estructura Atómica: Nanografeno (HBC fusionado) $C_{42}H_{18}$")
ax3.set_xlabel(r'$x$ [\AA]')
ax3.set_ylabel(r'$y$ [\AA]')
ax3.set_aspect('equal') # Fundamental para que los hexágonos no se deformen
ax3.grid(True, linestyle=':', alpha=0.5)
ax3.legend(loc='upper right')

if save: 
    plt.savefig('figures/estructura_atomica_nanografeno_fused.png', dpi=300)
    print("Figura de estructura guardada.")

plt.show()


# ==========================================
# 3. PATRÓN 2D NANOGRAFENO (HBC)
# ==========================================
print("\nGenerando Patrón 2D (Figura 3)...")
fig4, ax4 = plt.subplots(figsize=(10, 5))

# Crear un perfil transversal gaussiano para el haz en el eje Y
y_img = np.linspace(-3, 3, 400)
X_img, Y_img = np.meshgrid(x_screen, y_img)

# La intensidad 2D es la intensidad de interferencia en X * envolvente gaussiana en Y
# Modulamos la intensidad teórica calculada antes
envolvente_y = np.exp(-Y_img**2 / (2 * 0.8**2)) # sigma_y = 0.8
I_2D = total_intensity[np.newaxis, :] * envolvente_y

# Visualización usando imshow
# extent define los límites [xmin, xmax, ymin, ymax]
im = ax4.imshow(I_2D, extent=[-15, 15, -3, 3], cmap='viridis', aspect='auto', origin='lower')

ax4.set_title(r"Simulación del Patrón de Interferencia 2D en Pantalla")
ax4.set_xlabel(r'Posición horizontal $x$')
ax4.set_ylabel(r'Perfil transversal del haz $y$')

# Añadir barra de colores
cbar = plt.colorbar(im, ax=ax4, pad=0.02)
cbar.set_label('Intensidad Relativa')

if save: 
    plt.savefig('figures/2d_pattern_graphene.png', dpi=300)
    print("Figura de patrón 2D guardada.")

plt.show()


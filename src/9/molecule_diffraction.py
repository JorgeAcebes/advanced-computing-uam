# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Configuración de LaTeX con fuente LM Roman
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "text.latex.preamble": r"\usepackage{lmodern}"
})

save = 0
# ==========================================
# PARÁMETROS FÍSICOS: NANOGRAFENO (HBC)
# ==========================================
h_bar = 1.0 
mass = 12.0 * 42  # Masa del Carbono * 42 (Aprox. HBC)
v_most_probable = 0.5 
v_spread = 0.01   # Alta coherencia longitudinal necesaria para grafeno

grating_period = 2.5  # d
slit_width = 0.05     # a (Rendijas muy finas para dispersión amplia)
N_slits = 40          # Gran número de rendijas para picos ultra-definidos
L_screen = 30.0 

x_screen = np.linspace(-15, 15, 1200)
theta = np.arctan(x_screen / L_screen)
dx = x_screen[1] - x_screen[0]
bins_edges = np.append(x_screen, x_screen[-1] + dx)

# --- Cálculo de Intensidad (Lineal) ---
def diffraction_intensity(theta, lam, d, a, N):
    theta = np.where(theta == 0, 1e-12, theta)
    beta = (np.pi * a / lam) * np.sin(theta)
    delta = (2 * np.pi * d / lam) * np.sin(theta)
    # Factor de forma (difracción por rendija) * Factor de estructura (interferencia)
    return ((np.sin(beta) / beta)**2) * ((np.sin(N * delta / 2) / np.sin(delta / 2))**2)

# Espectro de velocidades
velocities = np.linspace(v_most_probable - 3*v_spread, v_most_probable + 3*v_spread, 40)
weights = np.exp(-0.5 * ((velocities - v_most_probable) / v_spread)**2)
weights /= np.sum(weights)

total_intensity = np.zeros_like(x_screen)
for v, w in zip(velocities, weights):
    lam = 2 * np.pi * h_bar / (mass * v)
    total_intensity += w * diffraction_intensity(theta, lam, grating_period, slit_width, N_slits)

total_intensity /= np.max(total_intensity)


# ==========================================
# 1. ANIMACIÓN (EJE LINEAL)
# ==========================================
fig2, (ax_anim, ax_hist) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={'height_ratios': [1.5, 1]})

ax_anim.set_xlim(-15, 15)
ax_anim.set_ylim(-5, L_screen + 2)
ax_anim.axhline(0, color='gray', lw=3, linestyle='--')
ax_anim.axhline(L_screen, color='black', lw=2)
ax_anim.set_xlabel(r'Posición transversal $x$')
ax_anim.set_ylabel(r'Distancia de propagación $z$')
scat = ax_anim.scatter([], [], s=4, c='lime', alpha=0.6)

# Configuración Eje Lineal
ax_hist.set_xlim(-15, 15)
ax_hist.set_ylim(0, 1.1)
ax_hist.set_ylabel(r"Intensidad $I/I_0$")
hist_line, = ax_hist.plot(x_screen, np.zeros_like(x_screen), color='lime', lw=1.5, label='Datos Experimentales')
ax_hist.fill_between(x_screen, 0, total_intensity, color='lime', alpha=0.1)
ax_hist.plot(x_screen, total_intensity, color='cyan', alpha=0.2, lw=0.3, label='Perfil Teórico')
ax_hist.legend()


n_particles = 1200
screen_hits = []

def update(frame):
    current_wave = frame % 60
    z_pos = -5 + (L_screen + 5) * (current_wave / 50.0)
    
    # Muestreo de la distribución de probabilidad
    batch_x = np.random.choice(x_screen, size=n_particles, p=total_intensity/np.sum(total_intensity))
    
    if z_pos < L_screen:
        x_pos = batch_x * (z_pos + 5) / (L_screen + 5)
        scat.set_offsets(np.c_[x_pos, np.full_like(x_pos, z_pos)])
    else:
        screen_hits.extend(batch_x)
        if len(screen_hits) > 0:
            hist, _ = np.histogram(screen_hits, bins=bins_edges)
            hist_line.set_ydata(hist / np.max(hist) if np.max(hist) > 0 else hist)
        scat.set_offsets(np.empty((0, 2)))
    return scat, hist_line

ani = FuncAnimation(fig2, update, frames=250, blit=True, interval=25)
if save: ani.save('figures/animation_reciprocal_space.mp4', dpi=300)
plt.show()

# ==========================================
# 2. REPRESENTACIÓN ESTRUCTURA GRAFÉNICA (HBC)
# ==========================================
fig3 = plt.figure(figsize=(8, 7.25))
ax3 = fig3.add_subplot(111)

def gen_hex(center, r):
    t = np.linspace(0, 2*np.pi, 7)
    return center[0] + r*np.cos(t), center[1] + r*np.sin(t), np.zeros_like(t)

# Generación de la celda central y las 6 periféricas (HBC)
r_bond = 1.42 # Distancia C-C en grafeno (Angstroms)
centers = [[0, 0]]
for i in range(6):
    angle = i * np.pi / 3
    centers.append([np.sqrt(3)*r_bond*np.cos(angle), np.sqrt(3)*r_bond*np.sin(angle)])

for c in centers:
    hx, hy, hz = gen_hex(c, r_bond/np.cos(np.pi/6)*0.5) # Ajuste visual de hexágonos
    ax3.plot(hx, hy, 'g-', lw=2)
    ax3.scatter(hx, hy, color='black', s=20, zorder=20)

ax3.set_title(r"Estructura Atómica: Nanografeno (HBC)")
ax3.set_xlabel(r'$x$ [\AA]')
ax3.set_ylabel(r'$y$ [\AA]')

if save: plt.savefig('figures/estructura_atomica_nanografeno.pdf', dpi=300)
plt.show()

# ==========================================
# 3. PATRÓN 2D NANOGRAFENO (HBC)
# ==========================================
fig4, ax4 = plt.subplots(figsize=(10, 4))
y_img = np.linspace(-3, 3, 400)
X_img, Y_img = np.meshgrid(x_screen, y_img)
I_2D = total_intensity * np.exp(-Y_img**2 / 0.8)
im = ax4.imshow(I_2D, extent=[-15, 15, -3, 3], cmap='viridis', aspect='auto')
ax4.set_title(r"Patrón de Interferencia de Nanografeno")
cbar = plt.colorbar(im, ax=ax4)
cbar.set_label('Intensidad de Interferencia')
ax4.set_xlabel(r'Posición sobre la pantalla')
ax4.set_ylabel(r'Perfil transversal del haz')
if save: plt.savefig('figures/2d_graphene.pdf', dpi=300)
plt.show()
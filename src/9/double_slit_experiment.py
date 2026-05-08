# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import hsv_to_rgb
from scipy.fft import fft2, ifft2, fftfreq
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# 2. Tipografía LaTeX (LM Roman)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.formatter.use_mathtext": True
})

# Parámetros físicos (unidades atómicas \hbar = m = e = 1)
Lx, Ly = 100.0, 100.0
Nx, Ny = 512, 512
dx, dy = Lx/Nx, Ly/Ny
dt = 0.05
steps_per_frame = 1

x = np.linspace(-Lx/2, Lx/2, Nx) 
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y) 

kx = fftfreq(Nx, d=dx) * 2 * np.pi 
ky = fftfreq(Ny, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky) 
K2 = KX**2 + KY**2

# Estado inicial (Paquete gaussiano)
x0, y0 = -30.0, 0.0
px, py = 5.0, 0.0
sigma = 2.0
psi = np.exp(-((X-x0)**2 + (Y-y0)**2)/(4*sigma**2)) * np.exp(1j*(px*X + py*Y))
psi /= np.sqrt(np.sum(np.abs(psi)**2)*dx*dy)

# Máximo inicial para escalar la amplitud de forma absoluta y evitar ruido por absorción
v_max_inicial = np.max(np.abs(psi))

# Potencial de la doble rendija
V = np.zeros((Ny, Nx))
slit_width, slit_sep, barrier_width, barrier_x = 2.0, 6.0, 2.0, 0.0
in_barrier_x = (X > barrier_x - barrier_width/2) & (X < barrier_x + barrier_width/2)
is_slit1 = (Y > slit_sep/2 - slit_width/2) & (Y < slit_sep/2 + slit_width/2)
is_slit2 = (Y > -slit_sep/2 - slit_width/2) & (Y < -slit_sep/2 + slit_width/2)
V[in_barrier_x & ~(is_slit1 | is_slit2)] = 1e5

# Capa absorbente
gamma, margin = 0.5, 10.0
absorber = np.where(np.abs(X) > Lx/2 - margin, gamma * (np.abs(X) - (Lx/2 - margin))**2, 0) + \
           np.where(np.abs(Y) > Ly/2 - margin, gamma * (np.abs(Y) - (Ly/2 - margin))**2, 0)
V_complex = V - 1j * absorber

# Operadores (Split-Step Fourier Method)
UR = np.exp(-1j * V_complex * dt / 2)
UK = np.exp(-1j * K2 * dt / 2)

def step(psi_in):
    psi_out = UR * psi_in
    psi_k = fft2(psi_out)
    psi_k = UK * psi_k
    psi_out = ifft2(psi_k)
    return UR * psi_out

def complex_to_rgba(Z):
    amp = np.abs(Z)
    H = (np.angle(Z) + np.pi) / (2 * np.pi)
    S = np.ones_like(H)
    # Acotamos a 1 usando el máximo inicial
    V_val = np.clip(amp / v_max_inicial, 0, 1) 
    return hsv_to_rgb(np.dstack((H, S, V_val)))

fig, ax = plt.subplots(figsize=(8,8))
img = ax.imshow(complex_to_rgba(psi), extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], origin='lower')

# El potencial es invariante, se dibuja solo una vez y no se actualiza en el loop
ax.contour(X, Y, V, levels=[5e4], colors='white', alpha=0.8, linewidths=1)

# 1. Unidades físicas a los ejes (Bohr radius a_0)
ax.set_xlabel(r'$x \ (\mathrm{a_0})$')
ax.set_ylabel(r'$y \ (\mathrm{a_0})$')
ax.set_title(r'Evolución de $\psi(x, y)$ - Doble Rendija')

# 3. Mostrar el colorbar para la fase
norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='hsv'), ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'$\arg(\psi)$')
cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

def update(frame):
    global psi
    for _ in range(steps_per_frame):
        psi = step(psi)
    
    img.set_data(complex_to_rgba(psi))
    # Devolvemos estrictamente el elemento dinámico
    return [img]

fps = 30
duracion_segundos = 10
total_frames = fps * duracion_segundos

ani = FuncAnimation(fig, update, frames=total_frames, blit=True)

ani.save('figures/doble_rendija.mp4', writer='ffmpeg', fps=fps, dpi=150)
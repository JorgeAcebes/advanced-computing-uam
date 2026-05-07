import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, eye, csc_matrix
from scipy.sparse.linalg import splu

# Configuración de LaTeX con fuente LM Roman
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{lmodern}"
})

# --- 1. PARÁMETROS BASE ---
L, dx = 1.0, 5e-4
x = np.arange(dx, L, dx)
N = len(x)
dt = 5e-7
k0 = 700.0
x0, sigma = 0.3, 0.03
xc = 0.6
w_base = 0.05
V0_base = 2.0 * k0**2

# --- 2. NÚCLEO FÍSICO Y SOLVER ---
def construir_sistema(V0, w):
    V = np.zeros(N)
    masc_barrera = (x >= xc - w/2) & (x <= xc + w/2)
    V[masc_barrera] = V0
    
    diag_prin = 1.0 / (dx**2) + V
    diag_sec = -0.5 / (dx**2) * np.ones(N - 1)
    H = diags([diag_sec, diag_prin, diag_sec], [-1, 0, 1])
    
    A = eye(N) + 0.5j * dt * H
    B = eye(N) - 0.5j * dt * H
    return B, splu(csc_matrix(A)).solve, V

def propagar_paquete(V0, w, t_max):
    psi_tmp = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi_tmp /= np.sqrt(np.trapezoid(np.abs(psi_tmp)**2, x))
    
    B_tmp, solve_A_tmp, _ = construir_sistema(V0, w)
    pasos = int(t_max / dt)
    
    for _ in range(pasos):
        psi_tmp = solve_A_tmp(B_tmp.dot(psi_tmp))
        
    masc_der = x > xc + w/2
    return np.trapezoid(np.abs(psi_tmp[masc_der])**2, x[masc_der])

# --- 3. ANIMACIÓN DE LA DINÁMICA ---
psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))

B, solve_A, V_actual = construir_sistema(V0_base, w_base)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.canvas.manager.set_window_title('Dinámica del Paquete de Ondas')

line_psi, = ax1.plot(x, np.abs(psi)**2, color='navy', label=r'$|\psi(x,t)|^2$')
ax1.fill_between(x, 0, V_actual / V0_base * np.max(np.abs(psi)**2), color='gray', alpha=0.3, label=r'$V(x)$')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, np.max(np.abs(psi)**2) * 1.5)
ax1.set_ylabel(r'Densidad de probabilidad $|\psi|^2$')
ax1.legend()

masc_der = x > xc + w_base/2
masc_izq = x < xc - w_base/2
time_data, T_data, R_data = [], [], []

line_T, = ax2.plot([], [], color='darkgreen', label=r'$T(t)$')
line_R, = ax2.plot([], [], color='darkred', label=r'$R(t)$')
ax2.set_xlim(0, 1.5e-3)
ax2.set_ylim(-0.05, 1.05)
ax2.set_xlabel(r'Tiempo $t$')
ax2.set_ylabel(r'Probabilidad')
ax2.legend()

t_actual = 0.0

def update(frame):
    global psi, t_actual
    for _ in range(50):
        psi = solve_A(B.dot(psi))
        t_actual += dt
        
    T = np.trapezoid(np.abs(psi[masc_der])**2, x[masc_der])
    R = np.trapezoid(np.abs(psi[masc_izq])**2, x[masc_izq])
    
    time_data.append(t_actual)
    T_data.append(T)
    R_data.append(R)
    
    line_psi.set_ydata(np.abs(psi)**2)
    line_T.set_data(time_data, T_data)
    line_R.set_data(time_data, R_data)
    return line_psi, line_T, line_R

ani = FuncAnimation(fig, update, frames=300, interval=30, blit=True)
plt.tight_layout()
print("Ejecutando animacion. Cierra la ventana para continuar con el analisis parametrico.")
plt.show()

# --- 4. EXPLORACIÓN DE LA TRANSMISIÓN (GRÁFICAS) ---
print("Calculando graficas de exploracion parametrica...")
t_simulacion = 1.5e-3

V0_vals = np.linspace(0.5 * V0_base, 2.0 * V0_base, 10)
T_V0 = [propagar_paquete(v, w_base, t_simulacion) for v in V0_vals]

w_vals = np.linspace(0.01, 0.1, 10)
T_w = [propagar_paquete(V0_base, w, t_simulacion) for w in w_vals]

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
fig2.canvas.manager.set_window_title('Exploracion Parametrica de la Transmision')

ax3.plot(V0_vals, T_V0, 'o-', color='teal')
ax3.set_xlabel(r'Altura de la barrera $V_0$')
ax3.set_ylabel(r'Coeficiente de Transmision $T$')
ax3.set_title(r'Variacion de $T$ con $V_0$ ($w$ constante)')
ax3.grid(True, linestyle=':')

ax4.plot(w_vals, T_w, 's-', color='purple')
ax4.set_xlabel(r'Anchura de la barrera $w$')
ax4.set_ylabel(r'Coeficiente de Transmision $T$')
ax4.set_title(r'Variacion de $T$ con $w$ ($V_0$ constante)')
ax4.grid(True, linestyle=':')

plt.tight_layout()
plt.show()
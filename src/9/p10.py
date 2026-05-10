# %%
import numpy as np                                      
import matplotlib.pyplot as plt                         
from matplotlib.animation import FuncAnimation, FFMpegWriter  
from scipy.linalg import solve_banded                   

base_size = 18
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

# ============================================================
# PARÁMETROS GLOBALES
# ============================================================

L   = 1.0                              
dx  = 5e-4                             
x   = np.arange(0, L + dx, dx)         
Nx  = len(x)                           

dt  = 5e-6                             
Nt  = 5000                             

x0    = 0.20                           
sigma = 0.025                          
k0    = 120                            
E     = k0**2 / 2                      

xc      = 0.60                         
V0_base = 1.2 * E                      
w_base  = 0.015                        

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def construir_potencial(V0, w):
    V = np.zeros(Nx)                                      
    mask = (x >= xc - w/2) & (x <= xc + w/2)    
    V[mask] = V0                                          
    return V

def inicializar_psi():
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
    psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))  
    psi[0]  = 0                                        
    psi[-1] = 0                                        
    return psi

def construir_matrices_CN(V):
    alpha = 1j * dt / (4 * dx**2)     

    diag_A  = np.ones(Nx,    dtype=complex)   
    diag_B  = np.ones(Nx,    dtype=complex)   
    upper_A = np.zeros(Nx-1, dtype=complex)   
    lower_A = np.zeros(Nx-1, dtype=complex)   
    upper_B = np.zeros(Nx-1, dtype=complex)   
    lower_B = np.zeros(Nx-1, dtype=complex)   

    for i in range(1, Nx-1):          
        diag_A[i]    =  1 + 2*alpha + 1j*dt*V[i]/2
        diag_B[i]    =  1 - 2*alpha - 1j*dt*V[i]/2
        upper_A[i]   = -alpha          
        lower_A[i-1] = -alpha          
        upper_B[i]   =  alpha          
        lower_B[i-1] =  alpha          

    ab = np.zeros((3, Nx), dtype=complex)
    ab[0, 1:]  = upper_A               
    ab[1, :]   = diag_A                
    ab[2, :-1] = lower_A              

    return ab, lower_B, diag_B, upper_B

def apply_B(psi, lower_B, diag_B, upper_B):
    result = np.zeros_like(psi, dtype=complex)     
    result[1:-1] = (
        lower_B[:-1] * psi[:-2]        
        + diag_B[1:-1] * psi[1:-1]    
        + upper_B[1:]  * psi[2:]       
    )
    return result

def T_analitica(E, V0, w):
    if E < V0:
        kappa = np.sqrt(2 * (V0 - E))             
        return 1 / (1 + (V0**2 * np.sinh(kappa * w)**2) / (4 * E * (V0 - E)))
    else:
        q = np.sqrt(2 * (E - V0))                  
        return 1 / (1 + (V0**2 * np.sin(q * w)**2)    / (4 * E * (E - V0)))

def evolucion_completa(V0, w, guardar_frames=False):
    V   = construir_potencial(V0, w)       
    psi = inicializar_psi()                
    ab, lB, dB, uB = construir_matrices_CN(V)  

    T_list, R_list, P_list = [], [], []    
    frames_prob, frames_real = [], []      
    times = []                             

    left    = x < (xc - w/2)                               
    barrier = (x >= xc - w/2) & (x <= xc + w/2)           
    right   = x > (xc + w/2)                               

    for n in range(Nt):                    
        b   = apply_B(psi, lB, dB, uB)    
        psi = solve_banded((1, 1), ab, b) 
        psi[0] = psi[-1] = 0             
        psi /= np.sqrt(np.trapezoid(np.abs(psi)**2, x))  

        if n % 20 == 0:                    
            prob = np.abs(psi)**2                                        

            T_list.append(np.trapezoid(prob[right],   x[right]))        
            R_list.append(np.trapezoid(prob[left],    x[left]))         
            P_list.append(np.trapezoid(prob[barrier], x[barrier]))      
            times.append(n * dt)                                        

            if guardar_frames:                     
                frames_prob.append(prob.copy())    
                frames_real.append(np.real(psi).copy())  

    return times, T_list, R_list, P_list, frames_prob, frames_real


# ============================================================
# Simulación base
# ============================================================

print("=" * 50)
print("① Simulación base")
print("=" * 50)

V_base = construir_potencial(V0_base, w_base)   

times, T_list, R_list, P_list, frames_prob, frames_real = evolucion_completa(
    V0_base, w_base, guardar_frames=True
)

T_arr = np.array(T_list)   
R_arr = np.array(R_list)
P_arr = np.array(P_list)

# ---------- Resultados numéricos ----------
print(f"\nTransmisión final   T  = {T_arr[-1]:.6f}")                         
print(f"Reflexión final     R  = {R_arr[-1]:.6f}")                           
print(f"T + R               = {T_arr[-1]+R_arr[-1]:.6f}")                    
print(f"T + R + P_barrera   = {T_arr[-1]+R_arr[-1]+P_arr[-1]:.6f}")         

# ---------- Resultado analítico ----------
T_teo = T_analitica(E, V0_base, w_base)   
R_teo = 1 - T_teo                          

print(f"\nEnergía       E  = {E:.3f}")
print(f"Potencial    V0  = {V0_base:.3f}")
print(f"Teoría        T  = {T_teo:.6f}")
print(f"Teoría        R  = {R_teo:.6f}")

# ---------- Comparación con máximo numérico ----------
idx_max = np.argmax(T_arr)         
T_max   = T_arr[idx_max]           

print(f"\nPrimer máximo numérico de T:")
print(f"  T_max            = {T_max:.6f}")
print(f"  R(T_max)         = {R_arr[idx_max]:.6f}")
print(f"  P_barrera(T_max) = {P_arr[idx_max]:.6f}")
print(f"  T+R+P            = {T_max+R_arr[idx_max]+P_arr[idx_max]:.6f}")
print(f"  t(T_max)         = {times[idx_max]:.4e}")
print(f"  T_teórica        = {T_teo:.6f}")
print(f"  Error relativo   = {abs(T_max-T_teo)/T_teo:.4%}")   

# ============================================================
# Animaciones
# ============================================================

print("\nGenerando animaciones...")
writer = FFMpegWriter(fps=25) 

# --- Animación de la densidad de probabilidad |ψ|² ---
fig1, ax1 = plt.subplots(figsize=(10, 5))
line1, = ax1.plot([], [], lw=2)                                   

# Eje secundario para la barrera (altura real)
ax1_v = ax1.twinx()
ax1_v.fill_between(x, 0, V_base, color='red', alpha=0.3)
ax1_v.set_ylim(0, V0_base * 1.2)
ax1_v.set_ylabel(r'$V(x)$')

txt1 = ax1.text(0.02, 0.85, '', transform=ax1.transAxes, fontsize=12) 
ax1.set(xlim=(0, 1), ylim=(0, np.max(frames_prob) * 1.2),
        xlabel='x', ylabel=r'$|\psi(x,t)|^2$',
        title='Densidad de probabilidad')
ax1.grid(True)

def update1(f):
    line1.set_data(x, frames_prob[f])                              
    txt1.set_text(rf"$t = {times[f]:.5e}$" + "\n" +
                  rf"$T = {T_list[f]:.4f}$" + "\n" +
                  rf"$R = {R_list[f]:.4f}$" + "\n" +
                  rf"$P_{{\mathrm{{barrera}}}} = {P_list[f]:.4f}$")
    return line1, txt1

ani1 = FuncAnimation(fig1, update1, frames=len(frames_prob), interval=30, blit=False)                      
ani1.save("densidad_probabilidad.mp4", writer=writer) 
plt.close(fig1)                                                   

# --- Animación de la parte real Re(ψ) ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
line2, = ax2.plot([], [], lw=2)

# Eje secundario para la barrera (altura real)
ax2_v = ax2.twinx()
ax2_v.fill_between(x, 0, V_base, color='red', alpha=0.3)
ax2_v.set_ylim(0, V0_base * 1.2)
ax2_v.set_ylabel(r'$V(x)$')

txt2 = ax2.text(0.02, 0.90, '', transform=ax2.transAxes, fontsize=12)
ax2.set(xlim=(0, 1), ylim=(-1.2, 1.2),
        xlabel='x', ylabel=r'$\mathrm{Re}(\psi)$',
        title='Parte real de la función de onda')
ax2.grid(True)

def update2(f):
    line2.set_data(x, frames_real[f])                              
    txt2.set_text(rf"$t = {times[f]:.5e}$")
    return line2, txt2

ani2 = FuncAnimation(fig2, update2, frames=len(frames_real), interval=30, blit=False)
ani2.save("parte_real.mp4", writer=writer)
plt.close(fig2)

print("Animaciones guardadas: densidad_probabilidad.mp4  |  parte_real.mp4")
# ============================================================
# T(t), R(t), P_barrera(t) — conservación de probabilidad
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(times, T_arr,             label='T(t)',           lw=2)   # Transmisión acumulada en la región derecha
plt.plot(times, R_arr,             label='R(t)',           lw=2)   # Reflexión acumulada en la región izquierda
plt.plot(times, P_arr,             label='P_barrera(t)',   lw=2)   # Probabilidad transitoria dentro de la barrera
plt.plot(times, T_arr+R_arr+P_arr, '--', label='T+R+P_barrera', lw=2)  # Suma total: debe mantenerse ≈ 1
plt.xlabel('Tiempo')
plt.ylabel('Probabilidad')
plt.title('Conservación de probabilidad')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('conservacion_probabilidad.pdf', bbox_inches='tight')  # Guarda en PDF para el informe
plt.show()

# ============================================================
# Barrido en V0
# ============================================================

print("\n" + "=" * 50)
print("④ Barrido en V0")
print("=" * 50)

V0_values = V0_base * np.linspace(0.9, 5, 9)
T_map_V0  = []                                      # Lista para almacenar T(t) de cada simulación

for V0 in V0_values:
    print(f"   V0 = {V0:.1f}…", end='', flush=True)
    _, T_v, _, _, _, _ = evolucion_completa(V0, w_base)   # Simula con este V0 y w fijo
    T_map_V0.append(T_v)                                    # Acumula la serie T(t) resultante
    print(" done")

T_map_V0 = np.array(T_map_V0)   # Convierte a matriz 2D: filas = valores de V0, columnas = instantes

# --- Mapa de calor T(V0, t) ---
plt.figure(figsize=(8, 5))
plt.imshow(T_map_V0, aspect='auto', origin='lower',
           extent=[0, Nt*dt, V0_values[0], V0_values[-1]])  # Ejes: tiempo en x, V0 en y
plt.colorbar(label='T(t)')
plt.xlabel('Tiempo')
plt.ylabel('$V_0$')
plt.title('Evolución $T(V_0, t)$')
plt.tight_layout()
plt.savefig('mapa_T_V0.pdf', bbox_inches='tight')
plt.show()

# --- Promedio temporal ± desviación estándar ---
T_mean_V0 = T_map_V0.mean(axis=1)   # Media temporal de T para cada valor de V0
T_std_V0  = T_map_V0.std(axis=1)    # Desviación estándar temporal (barra de error)

plt.figure(figsize=(7, 5))
plt.errorbar(V0_values, T_mean_V0, yerr=T_std_V0,
             marker='o', capsize=4, linestyle='-')    # Barras de error = variabilidad temporal de T
plt.xlabel('$V_0$')
plt.ylabel(r'$\langle T \rangle$ (promedio temporal)')
plt.title('Transmisión media vs altura de barrera')
plt.grid(True)
plt.tight_layout()
plt.savefig('T_medio_vs_V0.pdf', bbox_inches='tight')
plt.show()

# --- Sin barras de error ---
plt.figure(figsize=(7, 5))
plt.plot(V0_values, T_mean_V0, marker='o')            # Misma gráfica pero más limpia visualmente
plt.xlabel('$V_0$')
plt.ylabel(r'$\langle T \rangle$ (promedio temporal)')
plt.title('Transmisión media vs altura de barrera (sin σ)')
plt.grid(True)
plt.tight_layout()
plt.savefig('T_medio_vs_V0_noerror.pdf', bbox_inches='tight')
plt.show()

# ============================================================
# Barrido en w
# ============================================================

print("\n" + "=" * 50)
print("Barrido en w")
print("=" * 50)

w_values = w_base * np.linspace(0.2, 5, 9)   # 9 valores de w entre 0.5·w_base y 1.5·w_base
T_map_w  = []                                    # Lista para almacenar T(t) de cada simulación

for w in w_values:
    print(f"   w = {w:.5f}…", end='', flush=True)
    _, T_w, _, _, _, _ = evolucion_completa(V0_base, w)   # Simula con V0 fijo y este valor de w
    T_map_w.append(T_w)                                     # Acumula la serie T(t) resultante
    print(" done")

T_map_w = np.array(T_map_w)   # Convierte a matriz 2D: filas = valores de w, columnas = instantes

# --- Mapa de calor T(w, t) ---
plt.figure(figsize=(8, 5))
plt.imshow(T_map_w, aspect='auto', origin='lower',
           extent=[0, Nt*dt, w_values[0], w_values[-1]])   # Ejes: tiempo en x, w en y
plt.colorbar(label='T(t)')
plt.xlabel('Tiempo')
plt.ylabel('$w$')
plt.title('Evolución $T(w, t)$')
plt.tight_layout()
plt.savefig('mapa_T_w.pdf', bbox_inches='tight')
plt.show()

# --- Promedio temporal ± desviación estándar ---
T_mean_w = T_map_w.mean(axis=1)   # Media temporal de T para cada valor de w
T_std_w  = T_map_w.std(axis=1)    # Desviación estándar temporal

plt.figure(figsize=(7, 5))
plt.errorbar(w_values, T_mean_w, yerr=T_std_w,
             marker='o', capsize=4, linestyle='-')    # La fuerte caída exponencial con w es la firma del túnel
plt.xlabel('$w$')
plt.ylabel(r'$\langle T \rangle$ (promedio temporal)')
plt.title('Transmisión media vs anchura de barrera')
plt.grid(True)
plt.tight_layout()
plt.savefig('T_medio_vs_w.pdf', bbox_inches='tight')
plt.show()

# --- Sin barras de error ---
plt.figure(figsize=(7, 5))
plt.plot(w_values, T_mean_w, marker='o')
plt.xlabel('$w$')
plt.ylabel(r'$\langle T \rangle$ (promedio temporal)')
plt.title('Transmisión media vs anchura de barrera (sin σ)')
plt.grid(True)
plt.tight_layout()
plt.savefig('T_medio_vs_w_noerror.pdf', bbox_inches='tight')
plt.show()


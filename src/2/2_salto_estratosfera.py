# ========== DESCRIPCIÓN DEL PROBLEMA ========
# Persona que salta desde la estratosfera (en honor a Felix Baumgartner)

# Cuerpo de masa m (paracaidista + equipo)
# Altura inicial y_0
# Arrastre rho como función de la altura
# rho(y) = rho_0 * e^{-y/H}


# EDOS :

# dy/dt = -v
# dv/dt = g - 0.5 * rho(y) * C_d * A * v**2 / m


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
grav = sp.constants.g


# Algunas fuciones para plotear

plt.rcParams.update({
    "text.usetex": False,           
    "font.family": "serif",
    "mathtext.fontset": "cm",        
    "font.serif": ["DejaVu Serif"],
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15
})


# Método de Euler para ecuaciones diferenciales acopladas

def euler(f, g, t0, tf, step, y0=None, v0 = None):
    '''
    f: función diferencial 1
    g: función diferencial 2
    f: función diferencial 
    t_i: tiempo inicio
    t_f: tiempo final
    y0: valor inicial de la variable diferenciada en f
    z0: valor inicial de la variable diferenciada en g
    step: paso de timepo
    '''

    t = np.arange(t0, tf+step, step)
    
    y = np.zeros(len(t))
    if y0 is None: y0 = 0
    y[0] = y0


    v = np.zeros(len(t))
    if v0 is None: v0 = 0
    v[0] = v0

    i = 0
    for i in range(0, len(t)-1):
        if y[i] >= 0:
            # Implementación de Euler-Cromer, ligeramente más robusto que Euler
            # Mismo delta t, pero la energía se mantiene acotada
            # Básicamente, actualizamos primero velocidad y después posición
            v[i+1] = v[i] + step * g(t[i],y[i],v[i])
            y[i+1] = y[i] + step * f(t[i],y[i],v[i+1])
        elif (y[i] <0) & (y[i-1] >= 0):
            break
    
    idx = i+1
    return t[:idx],y[:idx],v[:idx]


# DEFINICIÓN DE VARIABLES

m = 110 
C_d = 0.9
A = 0.65
rho_0 = 1.225
H = 8500 
y_0 = 39000
v_0 = 0

# Funciones caída

def pos_caida(t,y,v):
    return -v

def vel_caida(t,y,v):
    rho = rho_0 * np.exp(-y/H)
    return grav - 0.5 * rho * C_d * A * v**2 / m

t,y,v = euler(pos_caida, vel_caida, t0=0, tf=1000, step=0.001, y0=y_0, v0 = v_0)


# Posiciones y velocidades finales
tf, yf, vf = t[-1], y[-1], v[-1]

# Velocidad máxima
v_max = np.max(v)
tv_max = t[np.argmax(v)]


# Gráficas

_, ax1 = plt.subplots(figsize= (10,6), tight_layout=True) 

ax1.plot(t, y, 'k-', label='Altura paracaidista', alpha =0.4)
ax1.plot(tf, yf, 'rx', label=rf'Aterrizaje ($t=${tf:.2f} s)')
ax1.set_xlabel("Tiempo [s]")
ax1.set_ylabel("Posición [m]")
titulo = 'Posición en función del tiempo'
ax1.legend()
ax1.set_title(titulo)
plt.show()


_, ax2 = plt.subplots(figsize= (10,6), tight_layout=True) 

ax2.plot(t, v, 'k-', label='Velocidad paracaidista', alpha =0.4)
ax2.plot(tf, vf, 'go', label=rf'Velocidad final ($v_f=${vf:.2f} m/s)')
ax2.plot(tv_max, v_max, 'y*', label=rf'Velocidad máxima ($v_\max=${v_max:.2f} m/s)')
ax2.set_xlabel("Tiempo [s]")
ax2.set_ylabel("Velocidad [m/s]")
titulo = 'Velocidad en función del tiempo'
ax2.legend()
ax2.set_title(titulo)
plt.show()

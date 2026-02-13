import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
g = sp.constants.g



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


# Método de Euler implementado mediante función

def euler(f, t0, tf, step, y0=None ):
    '''
    f: función diferencial 
    t_i: tiempo inicio
    t_f: tiempo final
    y0: valor inicial 
    step: paso de timepo
    '''
    t = np.arange(t0, tf+step, step)
    
    y = np.zeros(len(t))
    if y0 is None: y0 = 0

    y[0] = y0

    for i in range(0, len(t)-1):
        y[i+1] = y[i] + step * f(t[i],y[i])
    
    return t,y



def caida_con_fricción(t,y):
    a = 10 
    b = 1
    return a - b*y


# ========================  PARA UN SOLO VALOR DE Delta t  ================================

# Se considera velocidad inciial = 0

t,y = euler(caida_con_fricción,0, 10, 0.1)


_, ax = plt.subplots(figsize= (10,6), tight_layout=True) 

ax.axhline(10,  color = "red", linewidth = 1, linestyle = "dashed", label = "Velocidad terminal alcanzada")
ax.plot(t, y, 'k-', label='Velocidad paracaidista', alpha =0.4)
ax.set_xlabel("Tiempo [uds. tiempo]")
ax.set_ylabel("Velocidad vertical [m/s]")
titulo = 'Velocidad en función del tiempo'
ax.legend()
ax.set_title(titulo)






# ==================== EULER PARA VARIOS VALORES DE Delta t  ================================

steps_vals = [0.01, 0.1, 0.25, 0.5, 1]
colores = ["k", "g", "b", "y", "pink"]

_, ax = plt.subplots(figsize= (10,6), tight_layout=True) 

for i in range(len(steps_vals)):
    t,y = euler(caida_con_fricción, 0, 10, steps_vals[i])
    ax.plot(t, y, c =colores[i], alpha =0.8,label = rf"$\Delta t$: {steps_vals[i]:.2f}")



ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Velocidad vertical [m/s]")
titulo = r'Velocidad en función del tiempo para varios $\Delta t$'
ax.set_title(titulo)
ax.legend()
plt.show()

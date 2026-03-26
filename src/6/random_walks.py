# %% Imports y Declaraciones
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy as sp
from scipy.stats import linregress
import sys
from pathlib import Path
import importlib
from matplotlib.ticker import MaxNLocator

figuras = Path(__file__).resolve().parent.parent / '6' / 'figures'
figuras.mkdir(parents=True, exist_ok=True)  

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
import tools as ja
importlib.reload(ja)
ja.setup_style(base_size=19, dpi=120)


# === Parámetros libres del código ===
one_dim = 0
two_dim_pols = 0
two_dim_cart = 0
three_dim = 0

bucles_for = 0
testing = 0

animate = 1
save_figs = 1

# ____________

# Función auxiliar para que el módulo de la posición nunca sea negativo en polares (sumando +pi al ángulo)
def fix_neg_r(r, theta):
    theta = np.where(r<0, theta+np.pi, theta)
    r = np.abs(r)

    return r, theta


# Valores y funciones
np.random.seed(42)

n_steps = 1000 # Número de pasos
n_walkers = 1000 if not testing else 100 # Número de caminantes

t = np.arange(0, n_steps, 1) # Intervalo temporal


# Paseo aleatorio en 1D
if one_dim:
    # Inicialización de las matrices de posición y de <r^2>
    x = np.zeros([n_steps, n_walkers])
    x2 = np.zeros_like(x)

    if bucles_for:    
        for i in np.arange(n_steps-1):
            rand = np.random.choice([-1, 1], size= n_walkers)
            x[i+1, :] = x[i, :] + rand 
            x2[i+1, :] = x[i+1]**2     # Cuadrado de la posición

    # Empleamos suma cumulativa para evitar el uso de bucles for
    else:
        rand = np.random.choice([-1, 1], size =(n_steps, n_walkers))
        x = np.cumsum(rand, axis =0)
        x2 = x**2
        
    # --- Gráficas 1D ---
    fig, ax = plt.subplots(1,2, figsize=(20, 6))
    
    for i in np.arange(n_walkers):
        ax[0].plot(t, x[:, i])

    x2_sum = np.sum(x2, axis =1)/n_walkers
    linear_reg= linregress(t,x2_sum)
    slope = linear_reg[0]
    const = linear_reg[1]
    sign = '+' if const > 0 else ''
    ax[1].plot(t, x2_sum, label='Simulación', color = 'violet')
    ax[1].plot(t, slope*t+const, label=rf'Ajuste Lineal: $\langle x^2 \rangle = {slope:.2f}  t {sign} {const:.2f}$', color = 'lawngreen')
    ax[1].legend()

    ax[0].set_xlabel("Tiempo $t$")
    ax[0].set_ylabel("Posición $x$")
    ax[0].set_title(f"Random Walks de {n_walkers} Caminantes")

    ax[1].set_xlabel("Tiempo $t$")
    ax[1].set_ylabel(r"Cuadrado promedio de distancia $\langle r^2 \rangle$")
    ax[1].set_title(r"Evolución temporal de $\langle r^2 \rangle$")

    if save_figs:
        plt.savefig(figuras/f'rw_1d.png', bbox_inches='tight', dpi = 300) 
    plt.show()
 

# Paseo aleatorio en 2D (cartesianas)

# La estructura es idéntica al caso de 1D, salvo el uso de normalización en el dx, dy para
# que la distancia recorrida en cada iteración sea 1

if two_dim_cart:
    x = np.zeros([n_steps, 2, n_walkers]) 
    x2 = np.zeros([n_steps, n_walkers])

    isqr2 = 1 / np.sqrt(2) # Para que la distancia recorrida en cada paso sea 1

    if bucles_for:
        for i in np.arange(n_steps-1):
            rand = np.random.choice([-isqr2, isqr2], size= (2, n_walkers)) 
            x[i+1, :, :] = x[i, :, :] + rand 
            x2[i+1, :] = np.sum(x[i+1, :, :]**2, axis = 0)  # Cuadrado del módulo de la posición (r)
    else:
        rand = np.random.choice([-isqr2, isqr2], size= (n_steps, 2, n_walkers)).astype(float)
        x = np.cumsum(rand, axis = 0)
        x2 = np.sum(x[:,:,:]**2, axis = 1)


    fig, ax = plt.subplots(figsize=(12, 12))
    fig_2, ax_2 = plt.subplots(figsize=(12, 6))
    
    n_walkers_plot = 0

    for i in np.arange(n_walkers):
        ax.plot(x[:, 0, i], x[:, 1, i], alpha = 0.2)
        n_walkers_plot += 1

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"Random Walks de {n_walkers_plot} Caminantes")

    x2_sum = np.sum(x2, axis =1)/n_walkers
    linear_reg= linregress(t,x2_sum)
    slope = linear_reg[0]
    const = linear_reg[1]
    sign = '+' if const > 0 else ''

    ax_2.plot(t, x2_sum,  label='Simulación', color = 'violet')
    ax_2.plot(t, slope*t+const, label=rf'Ajuste Lineal: $\langle r^2\rangle = {slope:.2f}  t {sign} {const:.2f}$', color = 'lawngreen')

    ax_2.set_xlabel("Tiempo $t$")
    ax_2.set_ylabel(rf"Cuadrado promedio de distancia $\langle r^2 \rangle$")
    ax_2.set_title(rf"Evolución temporal de $\langle r^2 \rangle$")
    ax_2.legend()

        
    if save_figs:
        fig.savefig(figuras/f'rw_2d_cart.png', bbox_inches='tight', dpi = 300)
        fig_2.savefig(figuras/f'rw_2d_ev_cart.png', bbox_inches='tight', dpi = 300)


    if animate:
        fig_anim = plt.figure()
        ax_anim = fig_anim.add_subplot(111)
    
        idx = np.random.randint(0, n_walkers)
        
        # Fijar límites para que la cámara no salte
        lim = np.max(np.abs(x[:, :, idx])) * 1.1
        ax_anim.set_xlim(-lim, lim)
        ax_anim.set_ylim(-lim, lim)

        # Etiquetas de los ejes
        ax_anim.set_xlabel(r'$x$', labelpad=15)
        ax_anim.set_ylabel(r'$y$', labelpad=15)

        # Título dinámico
        titulo_anim = ax_anim.set_title(f"Caminante {idx} - Paso 0")

        line, = ax_anim.plot([], [], color='crimson', lw=2, alpha = 0.5)
        punto, = ax_anim.plot([], [], 'ro') 
        

        def init():
            line.set_data([], [])
            punto.set_data([], [])
            return line, punto

        def update(frame):
            x_c = x[:frame, 0, idx]
            y_c = x[:frame, 1, idx]
            
            line.set_data(x_c, y_c)
            
            if frame > 0:
                punto.set_data([x_c[-1]], [y_c[-1]])
                titulo_anim.set_text(f"Caminante {idx} - Paso {frame}")
            
            return line, punto

        ani = FuncAnimation(fig_anim, update, frames=n_steps, init_func=init, 
                            blit=False, interval=10, repeat=False)
        
        if save_figs:
            ani.save(figuras/f'anim_rw_2d_cart.mp4', dpi = 300)

        plt.show()

    plt.show()


       
# Paseo aleatorio en 2D (polares)

# Código muy similar al de 1D pero necesitando hacer uso de la función fix_neg_r

if two_dim_pols:
    x = np.zeros([n_steps, 2, n_walkers]) 
    x2 = np.zeros([n_steps, n_walkers])

    if bucles_for:
        for i in np.arange(n_steps-1):
            rand = np.random.choice([-1, 1], size= (2, n_walkers)) * [[1], [0.1]]
            x[i+1, :, :] = x[i, :, :] + rand 
            x2[i+1, :] = x[i+1, 0, :]**2     # Cuadrado de la posición (r)
    else:
        rand = np.random.choice([-1, 1], size= (n_steps, 2, n_walkers)).astype(float)
        rand[:,1,:] *= 0.1
        x = np.cumsum(rand, axis = 0)
        x2 = x[:,0,:]**2


    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    fig_2, ax_2 = plt.subplots(figsize=(12, 6))
    
    for i in np.arange(n_walkers):
        r, theta = fix_neg_r(x[:, 0, i], x[:, 1, i])
        ax.plot(theta, r, alpha = 0.1)
    
    ax.set_title(f"Random Walks de {n_walkers} Caminantes")

    x2_sum = np.sum(x2, axis =1)/n_walkers
    linear_reg= linregress(t,x2_sum)
    slope = linear_reg[0]
    const = linear_reg[1]
    sign = '+' if const > 0 else ''

    ax_2.plot(t, x2_sum,  label='Simulación', color = 'violet')
    ax_2.plot(t, slope*t+const, label=rf'Ajuste Lineal: $\langle r^2\rangle = {slope:.2f}  t {sign} {const:.2f}$', color = 'lawngreen')
    ax_2.legend()


    ax_2.set_xlabel("Tiempo $t$")
    ax_2.set_ylabel(r"Cuadrado promedio de distancia $\langle r^2 \rangle$")
    ax_2.set_title(r"Evolución temporal de $\langle r^2 \rangle$")
    
    if save_figs:
        fig.savefig(figuras/f'rw_2d_polar.png', bbox_inches='tight', dpi = 300)
        fig_2.savefig(figuras/f'rw_2d_ev_polar.png', bbox_inches='tight', dpi = 300)
    

    if animate:
        fig_anim, ax_anim = plt.subplots(subplot_kw={'projection': 'polar'})

        i =np.random.randint(0,n_walkers)# Caminante animado
        r, theta = fix_neg_r(x[:, 0, i], x[:, 1, i])
        ax_anim.set_ylim(0, np.max(r)*1.1)

        line, = ax_anim.plot([], [], color='red', lw=2, alpha = 0.5)
        punto, = ax_anim.plot([], [], 'o', color = 'crimson') # Marcador de posición actual

        ax_anim.set_title(f"Animación del Caminante {i} en Polares")

        # Función de inicialización
        def init():
            line.set_data([], [])
            punto.set_data([], [])
            return line, punto

        # Función de actualización (frame i corresponde al tiempo i)
        def update(frame):
            theta_data = theta[:frame]
            r_data = r[:frame]
            
            line.set_data(theta_data, r_data)
            if frame > 0:
                punto.set_data([theta_data[-1]], [r_data[-1]])
            
            return line, punto

        # Crear animación
        ani = FuncAnimation(fig_anim, update, frames=n_steps, init_func=init, 
                            blit=True, interval=10, repeat=False)
        
        if save_figs:
            ani.save(figuras/f'anim_rw_2d_polar.mp4', dpi = 300)

        plt.show()

    plt.show()


# Paseo aleatorio en 3D. Código idéntico a 2D cartesianas, pero con un dx, dy, dz = 1/sqrt(3)
if three_dim:
    x = np.zeros([n_steps, 3, n_walkers]) 
    x2 = np.zeros([n_steps, n_walkers])

    isqr3 = 1/np.sqrt(3) # Para que la distancia recorrida en cada paso sea 1

    if bucles_for:
        for i in np.arange(n_steps-1):
            rand = np.random.choice([-isqr3, isqr3], size= (3, n_walkers)) 
            x[i+1, :, :] = x[i, :, :] + rand 
            x2[i+1, :] = np.sum(x[i+1, :, :]**2, axis = 0)  # Cuadrado del módulo de la posición (r)
    else:
        rand = np.random.choice([-isqr3, isqr3], size= (n_steps, 3, n_walkers)).astype(float)
        x = np.cumsum(rand, axis = 0)
        x2 = np.sum(x[:,:,:]**2, axis = 1)


    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    fig_2, ax_2 = plt.subplots(figsize=(12, 6))
    
    n_walkers_plot = 0
    ax_2.plot(t, np.sum(x2, axis =1)/n_walkers)

    for i in np.arange(n_walkers):
        if i % 20 == 0:
            ax.plot(x[:, 0, i], x[:, 1, i], x[:, 2, i], alpha = 0.2)
            n_walkers_plot += 1

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_title(f"Random Walks de {n_walkers_plot} Caminantes")

    x2_sum = np.sum(x2, axis =1)/n_walkers
    linear_reg= linregress(t,x2_sum)
    slope = linear_reg[0]
    const = linear_reg[1]
    sign = '+' if const > 0 else ''

    ax_2.plot(t, x2_sum,  label='Simulación', color = 'violet')
    ax_2.plot(t, slope*t+const, label=rf'Ajuste Lineal: $\langle r^2\rangle = {slope:.2f}  t {sign} {const:.2f}$', color = 'lawngreen')

    ax_2.set_xlabel("Tiempo $t$")
    ax_2.set_ylabel(rf"Cuadrado promedio de distancia $\langle r^2 \rangle$")
    ax_2.set_title(rf"Evolución temporal de $\langle r^2 \rangle$")
    ax_2.legend()


    if save_figs:
        fig.savefig(figuras/f'rw_3d.png', bbox_inches='tight', dpi = 300)
        fig_2.savefig(figuras/f'rw_3d_ev.png', bbox_inches='tight', dpi = 300)

    if animate:
        fig_anim = plt.figure()
        ax_anim = fig_anim.add_subplot(111, projection='3d')
    
        idx = np.random.randint(0, n_walkers)
        
        # Fijar límites para que la cámara no salte
        lim = np.max(np.abs(x[:, :, idx])) * 1.1
        ax_anim.set_xlim(-lim, lim)
        ax_anim.set_ylim(-lim, lim)
        ax_anim.set_zlim(-lim, lim)

        ax_anim.view_init(elev=20, azim=0)

        # Etiquetas de los ejes
        ax_anim.set_xlabel(r'$x$', labelpad=15)
        ax_anim.set_ylabel(r'$y$', labelpad=15)
        ax_anim.set_zlabel(r'$z$', labelpad=15)

        # Título dinámico
        titulo_anim = ax_anim.set_title(f"Caminante {idx} - Paso 0")

        line, = ax_anim.plot([], [], [], color='crimson', lw=2, alpha = 0.5)
        punto, = ax_anim.plot([], [], [], 'ro') 
        

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            punto.set_data([], [])
            punto.set_3d_properties([])
            return line, punto

        def update(frame):
            x_c = x[:frame, 0, idx]
            y_c = x[:frame, 1, idx]
            z_c = x[:frame, 2, idx]
            
            line.set_data(x_c, y_c)
            line.set_3d_properties(z_c)
            
            if frame > 0:
                punto.set_data([x_c[-1]], [y_c[-1]])
                punto.set_3d_properties([z_c[-1]])
                titulo_anim.set_text(f"Caminante {idx} - Paso {frame}")
                velocidad_rotacion = 0.5 
                angulo_actual = frame * velocidad_rotacion
                ax_anim.view_init(elev=20, azim=angulo_actual)
            
            return line, punto

        ani = FuncAnimation(fig_anim, update, frames=n_steps, init_func=init, 
                            blit=False, interval=10, repeat=False)
        if save_figs:
            ani.save(figuras/f'anim_rw_3d.mp4',  dpi = 300)

        plt.show()

    plt.show()




# %% Imports y Declaraciones
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy as sp
import sys
from pathlib import Path
import importlib
from scipy.stats import linregress
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
import tools as ja
importlib.reload(ja)
ja.setup_style()

figuras = Path(__file__).resolve().parent.parent / '4' / 'figures'
figuras.mkdir(parents=True, exist_ok=True)  

# Definición de funciones y constantes:

# --- Funciones para integrar ---

def pend_NL_forz_fric(t,state, F_D, q):
    o = state[0]
    w = state[1]
    a = - np.sin(o) * g/l  - q * w + F_D * np.sin(Omeg * t)
    return np.array([w,a])

# --- Constantes y Variables ---
g = sp.constants.g
l = g #[m] 
q = 0.5
t_tot = 100
dt = 0.01
dang = 0.1 # grados
Omeg = 2/3
t = np.arange(0,t_tot,dt)


ang1 = np.degrees(0.2)
y1 = [np.radians(ang1), 0]

ang2 = ang1+dang
y2 = [np.radians(ang2), 0]

F_D_list = [0.5, 0.95, 1.2, 1.44]

save_title = ''

plot_animation =  0 # Decide si mostrar o no la animación 

# -- Elije si se realiza el apartado a o el apartado b --

apartado = 'b'

if apartado == 'a':
    q_list= np.ones(2)*q
    save_title = 'chaos_q_cte'
elif apartado == 'b':
    q_list = np.array([q, q+0.1])
    save_title = 'chaos_q_no_cte'
else:
    print('[Error] - Debes introducir un apartado valido.')

q1, q2 = q_list

# %% ESTE FRAGMENTO SOLO ES NECESARIO CORRERLO 1 VEZ, YA QUE GUARDAMOS EL RESULTADO
realizar_busqueda = 0

if realizar_busqueda:
    # == Buscamos la driving force que haga que lambda = 0 ==

    F_D_for_lambda = np.arange(1.42, 1.5, dt*0.1)

    tol = 0.001

    F_D_change = 0

    for i, F_D in enumerate(F_D_for_lambda):
        # Resolvemos las EDOS
        sol_1 = ja.rk4_solver(lambda t, y: pend_NL_forz_fric(t, y, F_D, q1), t, y1)
        o_1_cont, w_1 = sol_1.T
        o_1 = ja.restringir(o_1_cont)

        sol_2 = ja.rk4_solver(lambda t, y: pend_NL_forz_fric(t, y, F_D, q2), t, y2)
        o_2_cont, w_2 = sol_2.T
        o_2 = ja.restringir(o_2_cont)

        o2_o1 = np.abs(o_2_cont - o_1_cont)

        idx_est = int(30/dt)
        t_est = t[idx_est:]
        o2_o1_est = o2_o1[idx_est:]
        mask= o2_o1_est > 0

        res = linregress(t_est[mask], np.log(o2_o1_est[mask]))
        lambda_ = res.slope


        if np.abs(lambda_) < tol:
            F_D_change = F_D
            print(rf'Para F_D = {F_D}, lambda = 0.')
            break

    np.savetxt(f'{save_title}.txt', 
            [F_D_change], 
            comments='')

# %%  Desempaquetamos los resultados
F_D_change = np.loadtxt(f'{save_title}.txt', unpack=True)
F_D_list.append(F_D_change)
F_D_list.append(4)
L= len(F_D_list)

# == ANÁLISIS EXTENSO CAOS  ==

fig, ax = plt.subplots(L,5, figsize=(48, 6*L), dpi=300)


for i, F_D in enumerate(F_D_list):
    # Resolvemos las EDOS
    sol_1 = ja.rk4_solver(lambda t, y: pend_NL_forz_fric(t, y, F_D, q1), t, y1)
    o_1_cont, w_1 = sol_1.T
    o_1 = ja.restringir(o_1_cont)

    sol_2 = ja.rk4_solver(lambda t, y: pend_NL_forz_fric(t, y, F_D, q2), t, y2)
    o_2_cont, w_2 = sol_2.T
    o_2 = ja.restringir(o_2_cont)


    # Ángulo en función del tiempo
    ax[i,0].plot(t, o_1, label = rf'$\theta_0 = {ang1:.2f}^\circ$ $q = {q1}$', color = 'violet', alpha=0.5) 
    ax[i,0].plot(t, o_2, label = rf'$\theta_0 = {ang2:.2f}^\circ$ $q = {q2}$', color = 'lawngreen', alpha=0.5)

    ja.setup_ax(ax[i,0], title=rf'Ángulo en función del tiempo para $F_D =$ {F_D:.2f}', 
                xlabel=r'Tiempo [s]', ylabel=r'Ángulo $\theta$ [rad]', legend = True)
    ax[i,0].legend(loc='upper right')

    # Espacio de Fases
    ax[i,1].plot(o_1, w_1, label = rf'$\theta_0 = {ang1:.2f}^\circ$ $q = {q1}$', color = 'violet', alpha=0.5)
    ax[i,1].plot(o_2, w_2, label = rf'$\theta_0 = {ang2:.2f}^\circ$ $q = {q2}$', color = 'lawngreen', alpha=0.5)

    ja.setup_ax(ax[i,1], title=rf'Espacio de Fases para $F_D =$ {F_D:.2f}', 
                xlabel=r'Posición angular $\theta$ [rad]', ylabel=r'Velocidad angular $\omega$ [rad/s]', legend=True)   

    ax[i,1].legend(loc='upper right')

    # Exponente de Lyapunov
    o2_o1 = np.abs(o_2_cont - o_1_cont)
    ax[i,3].plot(t, np.log(o2_o1), color = 'deepskyblue', alpha=0.5) 

    idx_est = int(30/dt)
    t_est = t[idx_est:]
    o2_o1_est = o2_o1[idx_est:]
    mask= o2_o1_est > 0

    res = linregress(t_est[mask], np.log(o2_o1_est[mask]))
    lambda_ = res.slope
    C = res.intercept
    ax[i,3].plot(t_est, (t_est*lambda_ + C), color = 'mediumblue', alpha = 0.7, zorder=1, label=rf'Ajuste Lineal ($\lambda =$ {lambda_:.2f})')

    ja.setup_ax(ax[i,3], title=rf'Obtención del Exponente de Lyapunov para $F_D =$ {F_D:.2f}', 
                xlabel = 'Tiempo [s]', ylabel = r'$\ln{|\theta_1- \theta_2|}$', legend=True)

    # Potencia espectral
    N_1 = len(o_1_cont)
    G_1 = np.fft.rfft(o_1_cont - np.mean(o_1_cont) )
    PSD_1 = np.abs(G_1)**2 / N_1
    freqs_1 = np.fft.rfftfreq(N_1, d=dt)

    N_2 = len(o_2_cont)
    G_2 = np.fft.rfft(o_2_cont - np.mean(o_2_cont))
    PSD_2 = np.abs(G_2)**2 / N_2
    freqs_2 = np.fft.rfftfreq(N_2, d=dt)

    ax[i,4].semilogy(freqs_1, PSD_1,  label = rf'$\theta_0 = {ang1:.2f}^\circ$ $q = {q1}$', color = 'violet', alpha=0.5)
    ax[i,4].semilogy(freqs_2, PSD_2,  label = rf'$\theta_0 = {ang2:.2f}^\circ$ $q = {q2}$', color = 'lawngreen', alpha=0.5)
    

    ja.setup_ax(ax[i,4], title=rf'Espectro de potencias de $\theta(t)$ para $F_D = $ {F_D:.2f}', 
                ylabel = rf'Densidad Espectral de Potencia $S_\theta(f)$ [rad$^2$/Hz]', xlabel = r'$f$ [Hz]', legend=True)

    ax[i,4].set_xlim(0, 2)
    ax[i,4].legend(loc='upper right')
    
    # Sección de Poincaré
    o_1_p, w_1_p, _ = ja.sec_poincare_forz(o_1, w_1, Omeg, t)
    o_2_p, w_2_p, _ = ja.sec_poincare_forz(o_2, w_2, Omeg, t)


    ax[i,2].scatter(o_1_p, w_1_p, label = rf'$\theta_0 = {ang1:.2f}^\circ$ $q = {q1}$', color = 'violet', alpha=0.5)
    ax[i,2].scatter(o_2_p, w_2_p, label = rf'$\theta_0 = {ang2:.2f}^\circ$ $q = {q2}$', color = 'lawngreen', alpha=0.5)

    ja.setup_ax(ax[i,2], title=rf'Sección de Poincaré para $F_D =$ {F_D:.2f}', 
                xlabel=r'Posición angular $\theta$ [rad]', ylabel=r'Velocidad angular $\omega$ [rad/s]', legend=True)   

    ax[i,2].legend(loc='upper right')
# Lógica de animación. 

    if plot_animation:
        X1 =  l* np.sin(o_1)
        Y1 = -l * np.cos(o_1)

        X2 =  l* np.sin(o_2)
        Y2 = -l * np.cos(o_2)

        fig = plt.figure(figsize=(10, 10))
        ax_anim = fig.add_subplot()

        # Elementos gráficos a actualizar
        cuerda1, = ax_anim.plot([], [], 'm-', lw=2, alpha = 0.3) 
        cuerda2, = ax_anim.plot([], [], 'g-', lw=2, alpha = 0.3) 
        point_centro, = ax_anim.plot([], [], 'ko', ms=8, zorder = 10)
        line1, = ax_anim.plot([], [], 'm--', lw=1, alpha=0.5, label=rf'$\theta_0 = {ang1:.2f}^\circ, q = {q1}$')
        point1, = ax_anim.plot([], [], 'mo', ms=8, zorder = 10)
        line2, = ax_anim.plot([], [], 'g--', lw=1, alpha=0.5, label=rf'$\theta_0 = {ang2:.2f}^\circ, q = {q2}$')
        point2, = ax_anim.plot([], [], 'go', ms=8, zorder = 10)

        extra = 1.15
        ax_anim.set_xlim(-l*extra,l*extra)
        ax_anim.set_ylim(-l*extra, l*extra)
        ax_anim.set_aspect('equal')
        ax_anim.set_xlabel('X [m]')
        ax_anim.set_ylabel('Y [m]')
        ax_anim.set_title(rf'Trayectoria Péndulo para $F_D$ = {F_D:.2f}')
        ax_anim.legend(loc='upper right')

        def update(frame):
            cuerda1.set_data([0, X1[frame]], [0, Y1[frame]])
            cuerda2.set_data([0, X2[frame]], [0, Y2[frame]])
            line1.set_data(X1[:frame], Y1[:frame])          
            line2.set_data(X2[:frame], Y2[:frame])          
            point1.set_data([X1[frame]], [Y1[frame]]) # Requiere lista o array
            point2.set_data([X2[frame]], [Y2[frame]]) # Requiere lista o array
            point_centro.set_data([0],[0])
            
            return line1, point1, line2, point2, cuerda1, cuerda2, point_centro

        # Crear animación
        ani = FuncAnimation(fig, update, frames=range(0,len(t), 10), interval=20, blit=False)

        # Para guardar: 
        ani.save(figuras/f'pendulo_{save_title}_{F_D:.2f}.mp4', fps=30)
        plt.show()


plt.savefig(figuras/f'{save_title}.png', bbox_inches='tight') # Formato vectorial preferido
plt.tight_layout()
plt.show()


# %% Poincare Extenso

def Poincare_extenso(F_D_list, q= 0.5, t_fin=7000):
    if isinstance(F_D_list, (float, int)):
        F_D_list = [F_D_list]

    if t_fin > 1000:
        print('Este proceso puede ser lento.')

    for F_D in F_D_list:
        t = np.arange(0,t_fin, dt)
        sol_1 = ja.rk4_solver(lambda t, y: pend_NL_forz_fric(t, y, F_D, q), t, y1)
        o_1, w_1 = sol_1.T
        o_1 = ja.restringir(o_1)

        sol_2 = ja.rk4_solver(lambda t, y: pend_NL_forz_fric(t, y, F_D, q), t, y2)
        o_2, w_2 = sol_2.T
        o_2 = ja.restringir(o_2)

        o_1, w_1, _ = ja.sec_poincare_forz(o_1, w_1, Omeg, t)
        o_2, w_2, _ = ja.sec_poincare_forz(o_2, w_2, Omeg, t)



        fig, ax = plt.subplots(figsize =(12,8), dpi=300)
        ax.scatter(o_1, w_1, label = rf'$\theta = {ang1}$', color = 'violet', alpha=0.5)
        ax.scatter(o_2, w_2, label = rf'$\theta = {ang2}$', color = 'lawngreen', alpha=0.5)


        ja.setup_ax(ax, title=rf'Sección de Poincaré para $F_D =$ {F_D}', 
                xlabel=r'Posición angular $\theta$ [rad]', ylabel=r'Velocidad angular $\omega$ [rad/s]')   
        

        # --- Zoom en región de interés ---
        ax_ins = inset_axes(ax, width="30%", height="30%", loc='upper right', borderpad=2)

        ax_ins.scatter(o_1, w_1, label = rf'$\theta = {ang1}$', color = 'violet', alpha=0.5)

        # Ajustar límites del inset para el zoom
        ax_ins.set_xlim(2, 3)
        ax_ins.set_ylim(-1.25, -0.8)
        ax_ins.tick_params(labelsize=8)

        # Dibujar líneas conectoras entre el zoom y el eje original
        mark_inset(ax, ax_ins, loc1=2, loc2=4, ec="0.8")
        plt.savefig(figuras/f'poincare_extenso.png', bbox_inches='tight') # Formato vectorial preferido
        plt.show()


# Poincare_extenso(1.2)


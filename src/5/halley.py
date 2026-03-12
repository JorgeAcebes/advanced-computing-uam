# %% Imports y Declaraciones
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy as sp
from scipy.stats import linregress
import sys
from pathlib import Path
import importlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

figuras = Path(__file__).resolve().parent.parent / '5' / 'figures'
figuras.mkdir(parents=True, exist_ok=True)  

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
import tools as ja
importlib.reload(ja)
ja.setup_style(base_size=19, dpi=120)

# === Decidir en qué modo (prueba o final), qué apartados y si animar o no ===
testing = 0
apartado = [1, 0, 0] # a+b, c, d
animate = 0
save_figs = 0

n_vueltas = 3

# --- Variables y Funciones ---
# Masas en unidades de masas solares * 4 pi^2
GM_S = 4 * np.pi ** 2
GM_T = GM_S / 332946
GM_J = GM_S * 0.00095479194
GM_h = 0
GM_P = 6.6e-6 * GM_S

# Distancias al Sol en UA
r_T = 1.0 # Distancia Tierra-Sol 
r_J = 5.2 # Distancia Jupiter-Sol
r_h = 0.59 # Perihelio Halley
p_P = 29.7 # Periehelio Plutón
a_P = 49.3 # Afelio Plutón
r_P = 39.5 # Distancia media Plutón

r_S = 0

# Periodos orbitales
T_T = 1.0
T_J = 11.86
T_h = 76
T_P = 39.5**(3/2) # Semieje mayor ** 3/2 (Kepler)

# Velocidades orbitales
v_T = 2 * np.pi * r_T / T_T
v_J = 2 * np.pi * r_J / T_J
v_h = ja.velocity_orbit(r_h, T_h)
v_P = ja.velocity_orbit(r_P, T_P)

# Intervalo temporal
t_tot = T_h*n_vueltas
t = np.arange(0, t_tot, 0.01 if testing else 0.001) # if no testing: dt = 8.7 horas 

# -- Ecuaciónes de movimiento --
def grav_sun(r):
    '''
    Ecuaciones diferenciales para la interacción gravitatoria entre un cuerpo y el Sol
    '''
    r_mod = np.linalg.norm(r)
    a = - GM_S * r / r_mod**3

    return a

def grav_sun_jup(r, mult):
    '''
    Ecuaciones diferenciales para la interacción gravitatoria del Sol sobre Júpiter y Halley + Júpitre sobre Halley

    r = ndarray x0_halley y0_halley x0_jupyter y0_jupyter

    mult = factor multiplicativo a la masa de Júpiter
    '''
    a = np.zeros_like(r)

    r_halley      = r[0:2]
    r_halley_mod  = np.linalg.norm(r_halley)
    r_jupyter     = r[2:4]
    r_jupyter_mod = np.linalg.norm(r_jupyter)

    # # También tenemos que calcular la distancia entre Halley y Júpiter
    r_h_r_J       = r_halley - r_jupyter
    r_h_r_J_mod   = np.linalg.norm(r_h_r_J)

    a[0:2] = - GM_S * r_halley  / r_halley_mod**3 - GM_J * mult * r_h_r_J / r_h_r_J_mod**3
    a[2:4] = - GM_S * r_jupyter / r_jupyter_mod**3

    return a

def precesion(r_sol, t):
    '''
    Calcula la precesión de un cuerpo.

    inputs:
        - r_sol (n, 2) [ndarray]
        - t            [ndarray]
    
    output:
        - theta        [ndarray] (en radianes)
        - transit      [ndarray] (tiempo en el que se alcanza el afelio) 
    '''
    
    theta = []
    transit = []

    r_mod = np.linalg.norm(r_sol, axis =1)
    long= len(t)

    for i in np.arange(long):
        if (i !=0) & (i != long-1):
            if (r_mod[i] > r_mod[i-1]) & (r_mod[i] > r_mod[i+1]):
                ang = np.arccos(r_sol[i, 0] / r_mod[i])

                theta.append(ang)
                transit.append(t[i])

    return np.array(theta), np.array(transit)



calcular = 1 # Legacy
r_sol, v_sol = ja.verlet_solver(grav_sun, t, np.array([r_h, 0]), np.array([0, v_h]))    

# === Apartado a) y b) ===

# Interacción gravitatoria del Sol sobre Halley:
if apartado[0]:
    afelio = np.max(np.linalg.norm(r_sol, axis = 1))
    print('='*50)
    print(f" Velocidad Orbital máxima [UA/año]:     {np.max(np.linalg.norm(v_sol, axis = 1)):.4f}")
    print(f" Distancia máxima al Sol [UA]:          {afelio:.4f}")
    print('-'*50)
    print(" "*15, "Comparación Halley-Plutón")
    hal_plut = afelio > a_P
    resultado = "No" if hal_plut == 0 else "Sí"
    prop = afelio / r_P

    hal_plut_bis = afelio > p_P
    resultado_bis = "No" if hal_plut_bis == 0 else "Sí"
    print(f" ¿Llega más lejos Halley que Plutón?:   {resultado}")
    print(f" ¿Afelio Halley > perihelio Plutón?:    {resultado_bis}")
    print(f" Afelio Halley = {prop:.3f} * distancia media Plutón")
    print('='*50)


    fig, ax = plt.subplots(figsize=(29, 7.5), dpi=120)
    ax.plot(r_sol[:, 0], r_sol[:, 1], label = 'Halley', color = 'darkviolet') #Trayectoria Halley
    ax.plot(0,0, '*', ms = 15, label = 'Sol', color ='gold' )


    ja.setup_ax(ax, title=r'Órbitas del Sistema Solar',
                xlabel=r'$x$ [AU]', ylabel=r'$y$ [AU]', legend=True)

    if save_figs:
        plt.savefig(figuras/f'halley_sun.png', bbox_inches='tight', dpi = 300) # Formato vectorial preferido

    plt.tight_layout()
    plt.show()
    plt.close()

    if animate:
        ext = 1.2
        ja.animar_trayectorias(
            datos=[r_sol.T, np.zeros_like(r_sol.T)], 
            duracion=5.0, fps=30,
            archivo="figures/halley.mp4",
            title=r"Trayectoria cometa Halley",
            xlabel=r"$x$ [AU]", ylabel=r"$y$ [AU]", zlabel=r"$z$ [AU]",
            xlim = (np.min(r_sol[:, 0] - ext), np.max(r_sol[:, 0]) + ext),
            ylim = (np.min(r_sol[:, 1] * ext), np.max(r_sol[:, 1]) * ext),
            figsize = (29, 7.5),
            colors=['darkviolet', 'gold'], labels=['Halley', 'Sol'],
            guardar = True if save_figs else False, dpi = 150
        )



# === Apartado c) ===
if apartado[1]:
    E_pot = - GM_S / np.linalg.norm(r_sol, axis = 1)
    E_cin = 0.5 * np.linalg.norm(v_sol, axis = 1)**2 
    E_mec = E_pot + E_cin 
    L     = np.abs(r_sol[:, 0] * v_sol[:, 1] - r_sol[:, 1] * v_sol[:, 0])

    fig, ax = plt.subplots(1,2, figsize=(24, 7.5), dpi=120)
    ax[0].plot(t, E_pot, label = 'Energía potencial', color = 'violet') 
    ax[0].plot(t, E_cin, label = 'Energía cinética', color = 'lawngreen')
    ax[0].plot(t, E_mec, label = 'Energía mecánica', color = 'salmon')
    
    ja.setup_ax(ax[0], title=r'Evolución de Energía específica',
                xlabel=r'$t$ [año]', ylabel=r'$E/m$ [AU$^2$/año]', legend=True)


    ax[1].plot(t, L, label = r'$|\vec L|/m $')

    ja.setup_ax(ax[1], title=r'Evolución del Momento Angular específico',
                xlabel=r'$t$ [año]', ylabel=r'$|\vec L| / m$ [AU$^2$/año$^2$]', legend=True, 
                ylim = (5, 7))


    if save_figs:
        plt.savefig(figuras/f'energy_momentum.png', bbox_inches='tight', dpi = 300) # Formato vectorial preferido

    plt.tight_layout()
    plt.show()
    plt.close()


# === Apartado d) ===

'''
Se comentan las animaciones y los plots dado que al iterar sobre 50 factores multipicativos, no queremos infestar todo de plots.
'''

# Se emplea extrapolación
multiplicative = np.linspace(5, 11.5,100) # Tras inspección visual, se considera que 
# ese intervalo de factores multiplicativos logran que la órbita no se desestabilice
w = []
m_m = []
if apartado[2]:
    for factor_m_J in multiplicative:
        t = np.arange(0, t_tot, 0.01 if testing else 0.005)
        if calcular:
            r_sol_d, v_sol_d = ja.verlet_solver(lambda r: grav_sun_jup(r, factor_m_J), t, np.array([r_h, 0,  r_J, 0]), np.array([0, v_h, 0, -v_J]))    
            data_matrix = np.column_stack((r_sol_d[::1], t[::1]))

            theta, transit = precesion(r_sol_d, t)
            data_precesion = np.column_stack((theta, transit))

            np.savetxt('halley_and_jupyter.txt', 
                data_matrix, 
                header='x_h \t y_h \t x_J \t y_J \t t',
                comments='')
            
            np.savetxt('precesion.txt', 
                    data_precesion, 
                    comments='')

        #  Desempaquetamos los resultados y graficamos trayectoria
        x_h, y_h, x_J, y_J, t = np.loadtxt('halley_and_jupyter.txt', skiprows=1, unpack=True)    
        
        # fig, ax = plt.subplots(figsize=(29, 7.5), dpi=120)

        # ax.plot(x_h, y_h, label = 'Halley', color = 'darkviolet') #Trayectoria Halley
        # ax.plot(x_J, y_J, label = 'Júpiter', color = 'orangered') #Trayectoria Júpiter
        # ax.plot(0,0, '*', ms = 15, label = 'Sol', color ='gold' )
        
        # ja.setup_ax(ax, title=r'Órbitas del Sistema Solar',
        #             xlabel=r'$x$ [AU]', ylabel=r'$y$ [AU]', legend=True)

        # plt.tight_layout()
        # plt.show()
        # plt.close()

        # Desempaquetamos resultados y graficamos precesión:

        theta, transit = np.loadtxt('precesion.txt', unpack=True)
        if np.size(transit) > 1:
            res = linregress(transit, theta)

            m = res.slope
            n = res.intercept
            t_plot = np.linspace(0,t_tot, 50)
            
            w.append(m)
            m_m.append(factor_m_J)

            # fig, ax = plt.subplots(figsize=(12, 7.5), dpi=120)

        #     ax.plot(transit, theta, '-o', label = 'Precesión', color = 'darkviolet')
        #     ax.plot(t_plot, t_plot * m + n, color = 'cyan', label = rf'Ajuste Lineal: $\dot w=$ {m:.6f} rad/año')
            
        #     ja.setup_ax(ax, title=rf'Precesión Halley',
        #                 xlabel=r'Tiempo [año]', ylabel=r'Ángulo precesión [rad]', legend=True)
            
        # plt.tight_layout()
        # plt.show()
        # plt.close()


        # if animate and calcular:
        #     ext = 1.2
        #     ja.animar_trayectorias(
        #         datos=[np.array([x_h, y_h]), np.zeros_like(np.array([x_h, y_h])), np.array([x_J, y_J]) ], 
        #         duracion=10.0, fps=30, guardar=False, 
        #         archivo="figuras/halley_precesion.mp4",
        #         title=r"Trayectoria cometa Halley y Júpiter",
        #         xlabel=r"$x$ [AU]", ylabel=r"$y$ [AU]",
        #         xlim = (np.min(r_sol_d[:, 0] - ext), np.max(r_sol_d[:, 2]) + ext),
        #         ylim = (np.min(r_sol_d[:, 1] * ext), np.max(r_sol_d[:, 1]) * ext),
        #         figsize = (29, 7.5),
        #         colors=['darkviolet', 'gold', 'darkgoldenrod'], labels=['Halley', 'Sol', 'Júpiter'],
        #     )
    
    multiplicative = np.array(m_m)
    # Máscara trivial en la configuración actual, pero útil si se desea estudiar otros factores multiplicativos
    mask = (multiplicative > 5) & (multiplicative < 11.5)

    regress = linregress(multiplicative[mask], np.array(w)[mask] )
    pend = regress.slope
    C = regress.intercept
    m_plot = np.linspace(0.6,multiplicative[-1], 100)
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=120)
    ax.plot(multiplicative[mask], np.array(w)[mask], '-o', label = 'Precesión', color = 'darkviolet')
    ax.plot(m_plot, pend * m_plot + C, color = 'cyan', label = rf'Ajuste Lineal')
    ax.plot(1,  pend * 1 + C, 'r*', label = rf'$\dot w$ = {pend * 1 + C:.6f} rad/año')
    ax.set_xlabel('Factor multiplicativo masa de Júpiter')
    ax.set_ylabel('Precesión Halley [rad/año]')
    ax.set_title('Extrapolación Precesión Halley')
    plt.legend()

    if save_figs:
        plt.savefig(figuras/f'extrapolation.png', bbox_inches='tight', dpi = 300) # Formato vectorial preferido

    plt.show()

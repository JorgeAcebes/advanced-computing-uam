import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
g = sp.constants.g


# Parámetros de estilo

plt.rcParams.update({
    "text.usetex": False,           
    "font.family": "serif",
    "mathtext.fontset": "cm",        
    "font.serif": ["DejaVu Serif"],
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "figure.dpi": 150

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
        v[i+1] = v[i] + step * g(t[i],y[i],v[i])
        y[i+1] = y[i] + step * f(t[i],y[i],v[i])
    
    return t, y, v


# ================================================================

# Planteamiento del problema: Decaimiento de dos tipos de núcleos

# dNa/dt = -Na/ ta
# dNb/dt = Na/ta - Nb/tb


# Datos del problema
ta = 1 # tb/ta = k
k_vals = [0.2, 1.0, 20.0]
Na_0 = 100
Nb_0 = 0

t0 = 0
tf = 10
step = 0.01


# Resolución analítica:

import sympy as sp

t = sp.symbols('t', real=True, positive=True)
tau_A, tau_B, N_0 = sp.symbols('tau_A tau_B N_0', real=True, positive=True)
N_A = sp.Function('N_A')(t)
N_B = sp.Function('N_B')(t)

eq1 = sp.Eq(N_A.diff(t), -N_A / tau_A)
eq2 = sp.Eq(N_B.diff(t), (N_A / tau_A) - (N_B / tau_B))

sol = sp.dsolve([eq1, eq2], [N_A, N_B], ics={N_A.subs(t, 0): N_0, N_B.subs(t, 0): 0})

expr_nA = sol[0].rhs
expr_nB = sol[1].rhs

# Preparamos la función analítica para poder ser evaluada

f_nA = sp.lambdify((t, N_0, tau_A), expr_nA, modules='numpy')
f_nB = sp.lambdify((t, N_0, tau_A, tau_B), expr_nB, modules='numpy')


fig, axes = plt.subplots(1, len(k_vals), figsize=(14, 4), constrained_layout=True)


# Graficamos para distintos valores de tb
for i,k in enumerate(k_vals):
    tb = k

    # Ecuaciones diferenciales

    def dec_A(t,Na,Nb):
        return -Na/ta

    def dec_B(t,Na,Nb):
        return Na/ta - Nb/tb

    # Ejecución + Gráficas


    # Resolución Numérica
    t, Na, Nb = euler(dec_A, dec_B, t0, tf, step, y0 = Na_0, v0 = Nb_0)
    tf, Naf, Nbf = t[-1], Na[-1], Nb[-1]

    ax = axes[i]
    ax.plot(t, Na, 'b-', label=r'$N_a$', alpha =0.7)
    ax.plot(tf, Naf, 'b.')
    ax.plot(t, Nb, 'g-', label=r'$N_b$', alpha =0.7)
    ax.plot(tf, Nbf, 'g.')

    # Resolución Analítica
    val_tauB = k  

    # Ya sé que hay un divide by zero porque así es la función para Nb, pero no quiero que moleste en la terminal:
    with np.errstate(divide='ignore', invalid='ignore'):
        y_A = f_nA(t,  Na_0 , ta )
        y_B = f_nB(t,  Na_0 , ta , k)

    ax.plot(t,y_A, color = 'cyan', marker = "1", label=r'$N_a$ (analítica)', alpha = 0.7, zorder = 1);
    ax.plot(t,y_B, color = 'greenyellow', marker = "1", label=r'$N_b$ (analítica)', alpha = 0.7, zorder = 1);

    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Concentración [M]")
    titulo = rf'Relación: $\tau_b / \tau_a = {k}$'
    ax.legend()
    ax.set_title(titulo)

fig.suptitle(r"DECAIMIENTO RADIACTIVO", fontsize=16)
plt.show()

print('='*100 + '\nInteresante el caso tau_b/tau_a = 1.0.'
'\nLa solución numérica no es lo suficientemente precisa como para que se anule el'
'\ndenominador de la ecuación para N_b, mientras que la analítica sí.'
'\nEs por eso que en el segundo plot no vemos la curva analítica de N_b.\n' + '='*100)






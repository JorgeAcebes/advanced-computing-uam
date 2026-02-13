# %%
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import importlib

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
import tools as ja


importlib.reload(ja)


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


# --- Definimos los parámetros que vamos a emplear
v0 = 700 # [m/s] Velocidad inicial en módulo. 
B2m = 4e-5 # [1 / m] Constante del término cuadrático de la velocidad en Fuerza de arrastre
p0 = 1.225 # [kg/m**3] Densidad del aire a nivel del mar
g = sp.constants.g


a = 6.5e-3 #Valor de a en la fórmula adiabática
alpha = 2.5 # Exponente en la fórmula adibática
T0 = 293 #[K] Temperatura ambiente, para adibática


S = 4.1e-4 #[m] Para pelotas de Beisbol
wm = 200 # Velocidad angular entre la masa

# --- Valores iniciales e intervalo de tiempo ---

x0, y0 = 0, 0
vwx = 30 # [m/s] Velocidad del aire en la dirección x

t = np.linspace(0,1000,10000)

# -- Ecuaciones para resolver -- 
def caso_no_air(t, vec_ini):
    x, y = vec_ini[0], vec_ini[1]
    vx, vy = vec_ini[2], vec_ini[3]

#   Velocidades
    dx = vx 
    dy = vy

#   Aceleración
    dvx = 0
    dvy = -g

    return np.array([dx, dy, dvx, dvy])


def caso_only_roz(t,vec_ini):
    x, y = vec_ini[0], vec_ini[1]
    vx, vy = vec_ini[2], vec_ini[3]

#   Velocidades
    dx = vx 
    dy = vy

    v = np.sqrt(dx**2 + dy**2) # Calculo el módulo de la velocidad

#   Aceleración
    dvx = -B2m * v * dx
    dvy = -g - B2m * v * dy

    return np.array([dx, dy, dvx, dvy])


# Tanto para isotérmico como adiabático incorporo el efecto del viento

def caso_isotermo(t,vec_ini):

    x, y = vec_ini[0], vec_ini[1]
    vx, vy = vec_ini[2], vec_ini[3]

#   Velocidades
    dx = vx 
    dy = vy
    
    v = np.sqrt( (dx- vwx)**2 + dy**2) #Módulo de la velocidad real (quitandole la componente del viento)
    
    y0_scale = 8000.0  # Altura de escala
    densidad_factor = np.exp(-y / y0_scale) # La densidad baja al subir
    
    drag = B2m * densidad_factor * v

    dvx = -drag * (vx - vwx)
    dvy = -g - (drag * vy)

    return np.array([dx, dy, dvx, dvy])



def caso_adiab(t,vec_ini):

    x, y = vec_ini[0], vec_ini[1]
    vx, vy = vec_ini[2], vec_ini[3]

#   Velocidades
    dx = vx 
    dy = vy
    
    v = np.sqrt( (dx- vwx)**2 + dy**2) #Módulo de la velocidad real (quitandole la componente del viento)
    
    densidad_factor = (1 - a*y / T0) # La densidad baja al subir
    
    drag = B2m * densidad_factor * v 

    dvx = -drag * (vx - vwx)
    dvy = -g - (drag * vy)

    return np.array([dx, dy, dvx, dvy])


def caso_magnus(t, vec_ini):

    x, y = vec_ini[0], vec_ini[1]
    vx, vy = vec_ini[2], vec_ini[3]

#   Velocidades
    dx = vx 
    dy = vy
    
    v = np.sqrt( (dx- vwx)**2 + dy**2) #Módulo de la velocidad real (quitandole la componente del viento)

    dvx = -B2m * v * vx - S * wm * dy
    dvy = - g - B2m * v * vy + S * wm * dx

    return np.array([dx, dy, dvx, dvy])




theta = np.radians(45)  #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
vx0 = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)

inicio = [x0,y0,vx0,vy0]

sol_magnus = ja.euler_solver(caso_magnus, t, inicio, 1)
magnus = sol_magnus.T
t_magnus = t[:len(magnus[1])] # Truncamos el timpo para que tenga la misma dimensión que y


fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

ax.plot(magnus[0], magnus[1], '--', alpha=0.5, label=rf'Efecto Magnus', zorder=1)


ax.set_xlabel(r'Alcance $x$ [m]')
ax.set_ylabel(r'Altura $y$ [m]')
ax.set_title(r'Trayectoria Proyectil')

# Detalles finos (Grilla y Ticks)
ax.axhline(0, color='black', linewidth=0.8, alpha=0.8) # Eje cero
ax.grid(True, which='major', linestyle='-', alpha=0.6, color='#d3d3d3')
ax.minorticks_on() # Ticks menores son esenciales para lectura precisa
ax.grid(True, which='minor', linestyle=':', alpha=0.4)

# Leyenda (para eso sirve label)
ax.legend(frameon=True, loc='best', fontsize=12, framealpha=0.9, edgecolor='gray')

# Guardado y Muestra
plt.tight_layout()
# plt.savefig('plot_fisica.pdf', bbox_inches='tight') # Formato vectorial preferido
plt.show()
# plt.close()

# %%


# =================================================================================

# --- Variación de la trayectoria en función del ángulo --- 

fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

for angulo in np.arange(25, 75, 5):
    # Escribo las condiciones iniciales.

    theta = np.radians(angulo)  #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    inicio = [x0,y0,vx0,vy0]

    sol =  ja.euler_solver(caso_only_roz, t, inicio, 1)

    x, y, vx, vy = sol[:, 0], sol[:,1], sol[:,2], sol[:,3]

    t_sol = t[:len(y)] # Truncamos el timpo para que tenga la misma dimensión que y

    ax.plot(x, y, '--', alpha=0.5, label=rf'$\theta={angulo:.0f}$º', zorder=1)


# Formato de Ejes y Etiquetas
# Uso de raw strings (r'') para LaTeX dentro de mathtext
ax.set_xlabel(r'Alcance $x$ [m]')
ax.set_ylabel(r'Altura $y$ [m]')
ax.set_title(r'Trayectoria Proyectil - Con Aire')

# Detalles finos (Grilla y Ticks)
ax.axhline(0, color='black', linewidth=0.8, alpha=0.8) # Eje cero
ax.grid(True, which='major', linestyle='-', alpha=0.6, color='#d3d3d3')
ax.minorticks_on() # Ticks menores son esenciales para lectura precisa
ax.grid(True, which='minor', linestyle=':', alpha=0.4)

# Leyenda (para eso sirve label)
ax.legend(frameon=True, loc='best', fontsize=12, framealpha=0.9, edgecolor='gray')

# Guardado y Muestra
plt.tight_layout()
# plt.savefig('plot_fisica.pdf', bbox_inches='tight') # Formato vectorial preferido
plt.show()
# plt.close()


# =================================================================================

# %%

# --- Optimización no inteligente: Buscamos el mejor (mayor alcance) de todos los posibles ángulos

angle = np.arange(30,50,0.1)


# Inicializamos las variables donde guardaremos ángulo y alcance (x)
best_no_air = [0,0]
best_only_roz = [0,0]
best_isot = [0,0]
best_adiab = [0,0]

for ang in angle:    
    theta = np.radians(ang)  #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    inicio = [x0,y0,vx0,vy0]
    
    # -------------------------------------

    sol_no_air = ja.euler_solver(caso_no_air, t, inicio, 1)
    x_no_air = sol_no_air[:,0]


    if x_no_air[-1] > best_no_air[1]:
        best_no_air[0] = theta
        best_no_air[1] = x_no_air[-1]

    # -------------------------------------

    sol_only_roz = ja.euler_solver(caso_only_roz, t, inicio, 1)
    x_only_roz = sol_only_roz[:,0]

    if x_only_roz[-1] > best_only_roz[1]:
        best_only_roz[0] = theta
        best_only_roz[1] = x_only_roz[-1]

    # -------------------------------------

    sol_isot =  ja.euler_solver(caso_isotermo, t, inicio, 1)
    x_isot = sol_isot[:,0]

    if x_isot[-1] > best_isot[1]:
        best_isot[0] = theta
        best_isot[1] = x_isot[-1]

    # -------------------------------------

    sol_adiab =  ja.euler_solver(caso_adiab, t, inicio, 1)
    x_adiab = sol_adiab[:,0]

    if x_adiab[-1] > best_adiab[1]:
        best_adiab[0] = theta
        best_adiab[1] = x_adiab[-1]

print('='*40)

data = [
    ("Sin Aire", np.degrees(best_no_air[0]), best_no_air[1]),
    ("Con Aire", np.degrees(best_only_roz[0]), best_only_roz[1]),
    ("Isotérmico",np.degrees(best_isot[0]), best_isot[1]),
    ("Adiabático", np.degrees(best_adiab[0]), best_adiab[1]),
]

print(f"{'Modelo':<10} | {'Mejor Ángulo':>10} | {'Alcance':>10}")
print("-" * 40)

for modelo, ang, alcance in data:
    print(f"{modelo:<10} | {ang:>10.2f} | {alcance:>10.2f}")

# Guardamos los resultados
with open('best_resuts.dat', 'w') as f:
    for i in np.arange(len(data)):
            f.write('{} {:3.1f} {:.4f}\n'.format(data[i][0], data[i][1], data[i][2]))



# %%
# =================================================================================
# ______________ COMPARACIÓN DE LOS DISTINTOS MODELOS _________________



theta = np.radians(45)  #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
vx0 = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)

inicio = [x0,y0,vx0,vy0]

sol_no_air = ja.euler_solver(caso_no_air, t, inicio, 1)
no_air = sol_no_air.T
t_no_air = t[:len(no_air[1])] # Truncamos el timpo para que tenga la misma dimensión que y


sol_only_roz = ja.euler_solver(caso_only_roz, t, inicio, 1)
only_roz = sol_only_roz.T
t_only_roz = t[:len(only_roz[1])] # Truncamos el timpo para que tenga la misma dimensión que y


sol_isot =  ja.euler_solver(caso_isotermo, t, inicio, 1)
isot = sol_isot.T
t_isot = t[:len(isot[1])] # Truncamos el timpo para que tenga la misma dimensión que y

sol_adiab =  ja.euler_solver(caso_adiab, t, inicio, 1)
adiab = sol_adiab.T
t_adiab = t[:len(adiab[1])] # Truncamos el timpo para que tenga la misma dimensión que y



# A partir de aquí comienzan las gráficas
fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

ax.plot(no_air[0], no_air[1], '--', alpha=0.5, label=rf'Sin Aire', zorder=1)
ax.plot(only_roz[0], only_roz[1], '--', alpha=0.5, label=rf'Con aire', zorder=1)
ax.plot(isot[0], isot[1], '--', alpha=0.5, label=rf'Isotérmico', zorder=1)
ax.plot(adiab[0], adiab[1], '--', alpha=0.5, label=rf'Adiabático', zorder=1)


# Formato de Ejes y Etiquetas
# Uso de raw strings (r'') para LaTeX dentro de mathtext
ax.set_xlabel(r'Alcance $x$ [m]')
ax.set_ylabel(r'Altura $y$ [m]')
ax.set_title(r'Trayectoria Proyectil')

# Detalles finos (Grilla y Ticks)
ax.axhline(0, color='black', linewidth=0.8, alpha=0.8) # Eje cero
ax.grid(True, which='major', linestyle='-', alpha=0.6, color='#d3d3d3')
ax.minorticks_on() # Ticks menores son esenciales para lectura precisa
ax.grid(True, which='minor', linestyle=':', alpha=0.4)

# Leyenda (para eso sirve label)
ax.legend(frameon=True, loc='best', fontsize=12, framealpha=0.9, edgecolor='gray')

# Guardado y Muestra
plt.tight_layout()
# plt.savefig('plot_fisica.pdf', bbox_inches='tight') # Formato vectorial preferido
plt.show()
# plt.close()

# %%

# =================================================================================

# --- Variación de la trayectoria en función del ángulo --- 

fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

for angulo in np.arange(25, 75, 5):
    # Escribo las condiciones iniciales.

    theta = np.radians(angulo)  #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    inicio = [x0,y0,vx0,vy0]

    sol =  ja.euler_solver(caso_only_roz, t, inicio, 1)

    x, y, vx, vy = sol[:, 0], sol[:,1], sol[:,2], sol[:,3]

    t_sol = t[:len(y)] # Truncamos el timpo para que tenga la misma dimensión que y

    ax.plot(x, y, '--', alpha=0.5, label=rf'$\theta={angulo:.0f}$º', zorder=1)


# Formato de Ejes y Etiquetas
# Uso de raw strings (r'') para LaTeX dentro de mathtext
ax.set_xlabel(r'Alcance $x$ [m]')
ax.set_ylabel(r'Altura $y$ [m]')
ax.set_title(r'Trayectoria Proyectil - Con Aire')

# Detalles finos (Grilla y Ticks)
ax.axhline(0, color='black', linewidth=0.8, alpha=0.8) # Eje cero
ax.grid(True, which='major', linestyle='-', alpha=0.6, color='#d3d3d3')
ax.minorticks_on() # Ticks menores son esenciales para lectura precisa
ax.grid(True, which='minor', linestyle=':', alpha=0.4)

# Leyenda (para eso sirve label)
ax.legend(frameon=True, loc='best', fontsize=12, framealpha=0.9, edgecolor='gray')

# Guardado y Muestra
plt.tight_layout()
# plt.savefig('plot_fisica.pdf', bbox_inches='tight') # Formato vectorial preferido
plt.show()
# plt.close()



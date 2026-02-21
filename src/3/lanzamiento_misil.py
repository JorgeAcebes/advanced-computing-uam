# %% === IMPORTS y DEFINICIONES 
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

# Función para dar formato a los plots
def formato_lanzamiento(ax):
    ax.set_xlabel(r'Alcance $x$ [m]')
    ax.set_ylabel(r'Altura $y$ [m]')
    # Detalles finos (Grilla y Ticks)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.8) # Eje cero
    ax.grid(True, which='major', linestyle='-', alpha=0.6, color='#d3d3d3')
    ax.minorticks_on() # Ticks menores son esenciales para lectura precisa
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)

    # Leyenda (para eso sirve label)
    ax.legend(frameon=True, loc='best', fontsize=12, framealpha=0.6, edgecolor='gray')



# ==========================================================================================

#________________ PARÁMETROS Y FUNCIONES _______________________


# --- Definimos los parámetros que vamos a emplear
v0 = 700 # [m/s] Velocidad inicial en módulo. 
B2m = 4e-5 # [1 / m] Constante del término cuadrático de la velocidad en Fuerza de arrastre
p0 = 1.225 # [kg/m**3] Densidad del aire a nivel del mar
g = sp.constants.g
RT = 6371000 #[m] Radio terrestre

a = 6.5e-3 #Valor de a en la fórmula adiabática
alpha = 2.5 # Exponente en la fórmula adibática
T0 = 293 #[K] Temperatura ambiente, para adibática


S = 4.1e-4 #[m] Para pelotas de Beisbol
wm = 200 # Velocidad angular entre la masa. Vector en dirección z.

# --- Valores iniciales e intervalo de tiempo ---

x0, y0 = 0, 0
vwx = 100 # [m/s] Velocidad del viento en la dirección x

t = np.linspace(0,10000,10000)

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

def caso_no_air_g(t, vec_ini): #Gravedad Variable
    x, y = vec_ini[0], vec_ini[1]
    vx, vy = vec_ini[2], vec_ini[3]

#   Velocidades
    dx = vx 
    dy = vy

#   Aceleración
    dvx = 0
    dvy = -g *(RT / (RT + y)) ** 2 

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
    dvy = -g *(RT / (RT + y)) ** 2  - B2m * v * dy

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
    dvy = -g *(RT / (RT + y)) ** 2 - (drag * vy)

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
    dvy = -g *(RT / (RT + y)) ** 2 - (drag * vy)

    return np.array([dx, dy, dvx, dvy])

def caso_magnus(t, vec_ini):

    x, y = vec_ini[0], vec_ini[1]
    vx, vy = vec_ini[2], vec_ini[3]

#   Velocidades
    dx = vx 
    dx_rel = dx - vwx  #velocidad real (quitandole la componente del viento)
    dy = vy
    
    v = np.sqrt( dx_rel**2 + dy**2)

    dvx = -B2m * v * dx_rel - S * wm * dy
    dvy = -g *(RT / (RT + y)) ** 2 - B2m * v * dy + S * wm * dx_rel

    return np.array([dx, dy, dvx, dvy])



def magnus_3D(t, vec_ini):

    x, y, z = vec_ini[0], vec_ini[1], vec_ini[2]
    vx, vy, vz = vec_ini[3], vec_ini[4], vec_ini[5]

#   Velocidades
    dx = vx 
    dx_rel = dx - vwx #velocidad real (quitandole la componente del viento)
    dy = vy
    dz = vz
    
    v = np.sqrt(dx_rel**2 + dy**2 + dz**2) 

    dvx = -B2m * v * dx_rel - S * wm * dy
    dvy = -g *(RT / (RT + y)) ** 2 - B2m * v * dy + S * wm * dx_rel
    dvz =  -B2m * v * dz

    return np.array([dx, dy, dz, dvx, dvy, dvz])

# %% === Variación de la trayectoria en función del ángulo: Casos no aire y solo rozamiento ===
fig = plt.figure(figsize=(24,10), dpi=120)
gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[0, 1])

ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])



ang_range = np.arange(25, 80, 5)
diff_g = [] # Iniciailizamos vector diferencias altura g=cte vs g no cte
diff_a = [] # Ídem, para altura con y sin aire 

# Lo mismo para el alcance
delta_g = []
delta_a = []
 

for angulo in ang_range:
    # Escribo las condiciones iniciales.

    theta = np.radians(angulo)  #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    inicio = [x0,y0,vx0,vy0]


    sol_no_air =  ja.euler_solver(caso_no_air, t, inicio, 1)
    x_na, y_na, vx_na, vy_na = sol_no_air[:, 0], sol_no_air[:,1], sol_no_air[:,2], sol_no_air[:,3]
    t_sol_no_air = t[:len(y_na)] # Truncamos el timpo para que tenga la misma dimensión que y
    ax1.plot(x_na, y_na, '--', alpha=0.5, label=rf'$\theta={angulo:.0f}$º', zorder=1)

    sol_no_air_g =  ja.euler_solver(caso_no_air_g, t, inicio, 1)
    x_na_g, y_na_g, vx_na_g, vy_na_g = sol_no_air_g[:, 0], sol_no_air_g[:,1], sol_no_air_g[:,2], sol_no_air_g[:,3]
    t_sol_no_air_g = t[:len(y_na_g)] # Truncamos el timpo para que tenga la misma dimensión que y
    ax3.plot(x_na_g, y_na_g, '--', alpha=0.5, label=rf'$\theta={angulo:.0f}$º', zorder=1)


    sol =  ja.euler_solver(caso_only_roz, t, inicio, 1)
    x, y, vx, vy = sol[:, 0], sol[:,1], sol[:,2], sol[:,3]
    t_sol = t[:len(y)] # Truncamos el timpo para que tenga la misma dimensión que y
    ax2.plot(x, y, '--', alpha=0.5, label=rf'$\theta={angulo:.0f}$º', zorder=1)

    diff_g.append(max(y_na_g) - max(y_na))
    diff_a.append(max(y_na) - max(y))

    delta_g.append(max(x_na_g) - max(x_na))
    delta_a.append(max(x_na) - max(x))
    


ax4.plot(ang_range, diff_g, ':', alpha=0.8, label=rf'Altura máxima', zorder=1)
ax5.plot(ang_range, diff_a, ':', alpha=0.8, label='Altura máxima', zorder= 1)

ax4.plot(ang_range, delta_g, ':', alpha=0.8, label=rf'Alcance máximo', zorder=1)
ax5.plot(ang_range, delta_a, ':', alpha=0.8, label=rf'Alcance máximo', zorder=1)



# Formato de Ejes y Etiquetas

formato_lanzamiento(ax1)
formato_lanzamiento(ax2)
formato_lanzamiento(ax3)

ax1.set_title(r'Trayectoria Proyectil - Sin Aire')
ax2.set_title(r'Trayectoria Proyectil - Con Aire')
ax3.set_title(r'Trayectoria Proyectil - Sin Aire + Gravedad Variable')

ax4.set_xlabel(rf"Ángulo $\theta$")
ax4.set_ylabel("Diferencia de distancia [m]")
ax4.set_title(rf"$\Delta$ distancias máximas: $g \equiv$ cte vs $g = g(y)$")
ax4.legend()

ax5.set_xlabel(rf"Ángulo $\theta$")
ax5.set_ylabel("Diferencia de distancia [m]")
ax5.set_title(rf"$\Delta$ distancias máximas: Con Aire vs Sin Aire")
ax5.legend()

# Guardado y Muestra
plt.tight_layout()
# plt.savefig('plot_fisica.pdf', bbox_inches='tight') # Formato vectorial preferido
plt.show()
# plt.close()



print('='*60)
print('La diferencia es claramente notable en el caso Con y Sin Aire.')
print('Para el caso de gravedad cte vs variable, no es tan relevante.\nSin embargo, consideraremos a partir de ahora gravedad variable.')
print('='*60)

# %% === Comparación Distintos Modelos (Mismo Ángulo)

theta = np.radians(45) #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
inicio = [x0,y0, v0 * np.cos(theta), v0 * np.sin(theta)]

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

sol_magnus =  ja.euler_solver(caso_magnus, t, inicio, 1)
magnus = sol_magnus.T
t_magnus = t[:len(magnus[1])] # Truncamos el timpo para que tenga la misma dimensión que y


# A partir de aquí comienzan las gráficas
fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

ax.plot(no_air[0], no_air[1], '--', alpha=0.5, label=rf'Sin Aire', zorder=1)
ax.plot(only_roz[0], only_roz[1], '--', alpha=0.5, label=rf'Con aire', zorder=1)
ax.plot(isot[0], isot[1], '--', alpha=0.5, label=rf'Isotérmico', zorder=1)
ax.plot(adiab[0], adiab[1], '--', alpha=0.5, label=rf'Adiabático', zorder=1)
ax.plot(magnus[0], magnus[1], '--', alpha=0.5, label=rf'Magnus', zorder=1)



formato_lanzamiento(ax)
ax.set_title(rf'Trayectoria Proyectil - Ángulo $\theta = {np.degrees(theta):.0f}$')


# Guardado y Muestra
plt.tight_layout()
# plt.savefig('plot_fisica.pdf', bbox_inches='tight') # Formato vectorial preferido
plt.show()
# plt.close()

# %% === Efecto Magnus === 
theta = np.radians(45)  #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
vx0 = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)

inicio = [x0,y0,vx0,vy0]

sol_magnus = ja.euler_solver(caso_magnus, t, inicio, 1)
magnus = sol_magnus.T
t_magnus = t[:len(magnus[1])] # Truncamos el timpo para que tenga la misma dimensión que y


fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

ax.plot(magnus[0], magnus[1], '--', alpha=0.5, label=rf'Efecto Magnus + Viento', zorder=1)

formato_lanzamiento(ax)
ax.set_title(r'Trayectoria Proyectil')

# Guardado y Muestra
plt.tight_layout()
# plt.savefig('plot_fisica.pdf', bbox_inches='tight') # Formato vectorial preferido
plt.show()
# plt.close()


# %% === Magnus en 3D (Sin Animar) ===

theta = np.radians(19)  #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
vx0 = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)

z0 = 0
vz0 = 0
inicial_state = [x0,y0,z0,vx0,vy0,vz0]

sol_magnus_3D = ja.euler_solver(magnus_3D, t, inicial_state, 1)
mag_3D = sol_magnus_3D.T
t_magnus_3D = t[:len(mag_3D[1])]

fig = plt.figure(figsize=(10, 10), dpi=100)


ax = fig.add_subplot(projection='3d') # Hacemos un plot 3D 
ax.plot(mag_3D[0], mag_3D[2], mag_3D[1], color = 'violet', label='Magnus')

ax.set_xlabel('$x$ [m]', labelpad = 15)
ax.set_ylabel('$z$ [m]', labelpad = 15)
ax.set_zlabel('$y$ [m]', labelpad = 15)

ax.set_box_aspect((2, 1, 2)) # Escojo un tamaño de box particular (eje z más estrecho)

ax.set_title('Trayectoria con Efecto Magnus')

# plt.savefig('plot_fisica.pdf', bbox_inches='tight') # Formato vectorial preferido
plt.show()
plt.close()

# %% === Magnus en 3D (Animado) ===

import matplotlib.animation as animation

fig = plt.figure(figsize=(10, 8), dpi=100)
ax = fig.add_subplot(projection='3d')

line, = ax.plot([], [], [], lw=2, color='violet', label='Trayectoria') 
point, = ax.plot([], [], [], 'o', color='crimson', markersize=5) # El punto que está marcando el movimiento
shadow, = ax.plot([], [], [], '--', color='gray', alpha=0.5) # Proyección en el plano del suelo


ax.set_xlabel('$x$ [m]', labelpad = 15)
ax.set_ylabel('$z$ [m]', labelpad = 15)
ax.set_zlabel('$y$ [m]', labelpad = 15)


x_vals = mag_3D[0]
y_vals = mag_3D[1]
z_vals = mag_3D[2]

ax.set_xlim(0, np.max(x_vals))
ax.set_ylim(-0.2,0.2)
ax.set_zlim(0, np.max(y_vals))


# Para limitar el número de números que aparecen en los ejes
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax.zaxis.set_major_locator(MaxNLocator(nbins=5))

frames_totales = 400 # Número de frames que queremos ver
step = max(1, len(x_vals) // frames_totales)

def update(frame):
    # El índice real en el array de datos
    idx = frame * step
    if idx >= len(x_vals):
        idx = len(x_vals) - 1
        
    line.set_data(x_vals[:idx], z_vals[:idx])
    line.set_3d_properties(y_vals[:idx])
    
    point.set_data([x_vals[idx]], [z_vals[idx]]) 
    point.set_3d_properties([y_vals[idx]])

    shadow.set_data(x_vals[:idx], z_vals[:idx])
    shadow.set_3d_properties(np.zeros(idx))
    
    return line, point, shadow

ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=frames_totales, 
    interval=2, # milisegundos entre frames
    blit=False   # blit=False suele ser más estable en 3D
)

plt.show()

# %% === Optimización no inteligente: Buscamos el mejor (mayor alcance) de todos los posibles ángulos ===

angle = np.arange(0,50,0.1)

N = int(1e6)
t = np.linspace(0,N,N*10)

# Inicializamos las variables donde guardaremos ángulo y alcance (x)
best_no_air = [0,0]
best_only_roz = [0,0]
best_isot = [0,0]
best_adiab = [0,0]
best_magnus = [0,0]

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

    # -------------------------------------

    sol_magnus =  ja.euler_solver(caso_magnus, t, inicio, 1)
    x_magnus = sol_magnus[:,0]

    if x_magnus[-1] > best_magnus[1]:
        best_magnus[0] = theta
        best_magnus[1] = x_magnus[-1]

data = [
    ("No Air", np.degrees(best_no_air[0]), best_no_air[1]),
    ("With Air", np.degrees(best_only_roz[0]), best_only_roz[1]),
    ("Isothermic",np.degrees(best_isot[0]), best_isot[1]),
    ("Adiabatic", np.degrees(best_adiab[0]), best_adiab[1]),
    ("Magnus", np.degrees(best_magnus[0]), best_magnus[1]),
]

print('='*40)
print(f"{'Model':<10} | {'Best Angle':>10} | {'Reach':>10}")
print("-" * 40)

for modelo, ang, alcance in data:
    print(f"{modelo:<10} | {ang:>10.2f} | {alcance:>10.2f}")

print('='*40)

# Guardamos los mejores resultados en .txt

with open("best_results.txt", "w") as f:
    f.write(f"{'Model':<10}\t{'Angle':>10}\t{'Reach':>10}\n") 
    for modelo, ang, alcance in data:
        f.write(f"{modelo:<10}\t{ang:>10.0f}\t{alcance:>10.1f}\n")
print('Resultados Guardados')




#%% === COMPARACIÓN DE LOS DISTINTOS MODELOS (EN SU MEJOR ÁNGULO) ===

theta = best_no_air[0] #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
inicio = [x0,y0, v0 * np.cos(theta), v0 * np.sin(theta)]

sol_no_air = ja.euler_solver(caso_no_air, t, inicio, 1)
no_air = sol_no_air.T
t_no_air = t[:len(no_air[1])] # Truncamos el timpo para que tenga la misma dimensión que y


theta = best_only_roz[0] #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
inicio = [x0,y0, v0 * np.cos(theta), v0 * np.sin(theta)]

sol_only_roz = ja.euler_solver(caso_only_roz, t, inicio, 1)
only_roz = sol_only_roz.T
t_only_roz = t[:len(only_roz[1])] # Truncamos el timpo para que tenga la misma dimensión que y


theta = best_isot[0] #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
inicio = [x0,y0, v0 * np.cos(theta), v0 * np.sin(theta)]

sol_isot =  ja.euler_solver(caso_isotermo, t, inicio, 1)
isot = sol_isot.T
t_isot = t[:len(isot[1])] # Truncamos el timpo para que tenga la misma dimensión que y


theta = best_adiab[0] #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
inicio = [x0,y0, v0 * np.cos(theta), v0 * np.sin(theta)]

sol_adiab =  ja.euler_solver(caso_adiab, t, inicio, 1)
adiab = sol_adiab.T
t_adiab = t[:len(adiab[1])] # Truncamos el timpo para que tenga la misma dimensión que y


theta = best_magnus[0] #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
inicio = [x0,y0, v0 * np.cos(theta), v0 * np.sin(theta)]

sol_magnus =  ja.euler_solver(caso_magnus, t, inicio, 1)
magnus = sol_magnus.T
t_magnus = t[:len(magnus[1])] # Truncamos el timpo para que tenga la misma dimensión que y



# A partir de aquí comienzan las gráficas
fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

ax.plot(no_air[0], no_air[1], '--', alpha=0.5, label=rf'Sin Aire $\theta = {np.degrees(best_no_air[0]):.1f}$', zorder=1)
ax.plot(only_roz[0], only_roz[1], '--', alpha=0.5, label=rf'Con aire $\theta = {np.degrees(best_only_roz[0]):.1f}$', zorder=1)
ax.plot(isot[0], isot[1], '--', alpha=0.5, label=rf'Isotérmico $\theta = {np.degrees(best_isot[0]):.1f}$', zorder=1)
ax.plot(adiab[0], adiab[1], '--', alpha=0.5, label=rf'Adiabático $\theta = {np.degrees(best_adiab[0]):.1f}$', zorder=1)
ax.plot(magnus[0], magnus[1], '--', alpha=0.5, label=rf'Magnus $\theta = {np.degrees(best_magnus[0]):.1f}$', zorder=1)

formato_lanzamiento(ax)
ax.set_title(r'Trayectoria Proyectil - Ángulo Lanzamiento Óptimo')

plt.tight_layout()
# plt.savefig('plot_fisica.pdf', bbox_inches='tight') # Formato vectorial preferido
plt.show()
# plt.close()


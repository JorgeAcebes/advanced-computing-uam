# %% === IMPORTS y DEFINICIONES 
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import importlib
from pathlib import Path
from matplotlib.animation import FuncAnimation

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
import tools as ja
importlib.reload(ja)


# Configuración general de los plots
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


# Creamos un directorio para guardar figuras
figuras = Path("figures")
figuras.mkdir(parents=True, exist_ok=True)

# ==============================================================

#________________ PARÁMETROS Y FUNCIONES _______________________


# --- Definimos los parámetros que vamos a emplear
v0 = 700 # [m/s] Velocidad inicial en módulo. 
B2m = 4e-5 # [1 / m] Constante del término cuadrático de la velocidad en Fuerza de arrastre
p0 = 1.225 # [kg/m**3] Densidad del aire a nivel del mar
g = sp.constants.g
RT = 6371000 #[m] Radio terrestre
T0 = 288 #[K] Temperatura ambiente (T = 15º)

m = 4.81e-26 #[kg] Masa molécula de aire
kb = 1.38e-23 #[J/K] Constante de Boltzmann
# Altura de escala para isotérmica
y0_scale = kb * T0 / (m * g)

a = 6.5e-3 #Valor de a en la fórmula adiabática
alpha = 2.5 # Exponente en la fórmula adibática


S = 4.1e-4 #[m] Para pelotas de Beisbol
wm = 200 # Velocidad angular entre la masa. Vector en dirección z

# --- Valores iniciales e intervalo de tiempo ---

x0, y0 = 0, 0
vwx = 100 # [m/s] Velocidad del viento en la dirección x

t = np.linspace(0,1000,10000)

# --- Ecuaciones de movimiento a resolver ---
 
def caso_no_air(t, vec_ini): # Caso ideal
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
    dvy = -g * (RT / (RT + y)) ** 2 

    return np.array([dx, dy, dvx, dvy])

# A partir de aquí, todos tienen gravedad variable

def caso_only_roz(t,vec_ini): #Solo rozamiento 
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

def caso_isotermo(t,vec_ini): #Modelo isotermo

    x, y = vec_ini[0], vec_ini[1]
    vx, vy = vec_ini[2], vec_ini[3]

#   Velocidades
    dx = vx 
    dy = vy
    
    v = np.sqrt( (dx- vwx)**2 + dy**2) #Módulo de la velocidad real (quitandole la componente del viento)

    densidad_factor = np.exp(-y / y0_scale) # La densidad baja al subir
    
    drag = B2m * densidad_factor * v

    dvx = -drag * (vx - vwx)
    dvy = -g *(RT / (RT + y)) ** 2 - (drag * vy)

    return np.array([dx, dy, dvx, dvy])

def caso_adiab(t,vec_ini): # Modelo adiabático

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

def caso_magnus(t, vec_ini): # Modelo magnus

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


def magnus_3D(t, vec_ini): # Modelo Magnus en 3 dimensiones -- empleo notación vectorial

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



def magnus_3D_slice(t, vec_ini):

    x, y, z = vec_ini[0], vec_ini[1], vec_ini[2]
    vx, vy, vz = vec_ini[3], vec_ini[4], vec_ini[5]

#   Velocidades
    dx = vx 
    dx_rel = dx - vwx #velocidad real (quitandole la componente del viento)
    dy = vy
    dz = vz
    
    v = np.sqrt(dx_rel**2 + dy**2 + dz**2) 

    if v < 14:
        C = 1.0
    else:
        C = 14 / v
    
    rho = (1 - (a * z / T0)) ** alpha #En unidades de rho_0, de tal manera que no haya que dividir después por rho_0

    arrastre = C * rho * A_golf / m_golf

    dvx = - arrastre * v * dx_rel - S0wm * dy
    dvy = - arrastre * v * dy     + S0wm * dx_rel
    dvz = -g *(RT / (RT + z)) ** 2 - arrastre * v * dz 

    return np.array([dx, dy, dz, dvx, dvy, dvz])

def magnus_3D_hook(t, vec_ini):
    x, y, z = vec_ini[0], vec_ini[1], vec_ini[2]
    vx, vy, vz = vec_ini[3], vec_ini[4], vec_ini[5]

#   Velocidades
    dx = vx 
    dx_rel = dx - vwx #velocidad real (quitandole la componente del viento)
    dy = vy
    dz = vz
    
    v = np.sqrt(dx_rel**2 + dy**2 + dz**2) 

    if v < 14:
        C = 1.0
    else:
        C = 14 / v
    
    rho = (1 - (a * z / T0)) ** alpha

    arrastre = C * rho * A_golf / m_golf

    dvx = -arrastre * v * dx_rel + S0wm * dy
    dvy = -arrastre * v * dy     - S0wm * dx_rel
    dvz = -g *(RT / (RT + z)) ** 2 - arrastre * v * dz 

    return np.array([dx, dy, dz, dvx, dvy, dvz])

# ~~~ Funciones para jugar ~~~

def muelle_viscoelástico2(t, estado): # Implementación del modelo de Kelvin-Voig
    r1 = estado[0:3]
    v1 = estado[3:6]
    r2 = estado[6:9]
    v2 = estado[9:12]

    r12 = r1 - r2
    d12 = np.linalg.norm(r12) # Modulo de la distancia relativa
    u12 = r12 / (d12 + 1e-12) # Vector unitario en la dirección 1-2. Le sumo +10^-15 para evitar divisiones por 0

    v12 = v1 - v2

    F_viscelast =  - ( k * (d12 - L)  + c * np.dot(v12, u12) ) * u12
    a1 =  g + F_viscelast / m1 
    a2 =  g - F_viscelast / m2

    return np.concatenate((v1,a1,v2,a2))


def muelle_viscoelástico3(t, estado): # Implementación del modelo de Kelvin-Voig
    r1 = estado[0:3]
    v1 = estado[3:6]
    r2 = estado[6:9]
    v2 = estado[9:12]
    r3 = estado[12:15]
    v3 = estado[15:18]

    r12 = r1 - r2
    d12 = np.linalg.norm(r12) # Modulo de la distancia relativa
    u12 = r12 / (d12 + 1e-12) # Vector unitario en la dirección 1-2. Le sumo +10^-15 para evitar divisiones por 0
    v12 = v1 - v2

    r13 = r1 - r3
    d13 = np.linalg.norm(r13)
    u13 = r13/ (d13 + 1e-12)
    v13 = v1 - v3

    r23 = r2 - r3
    d23 = np.linalg.norm(r23)
    u23 = r23 / (d23 + 1e-12)
    v23 = v2 - v3

    F_viscelast_12 =  - ( k * (d12 - L)  + c * np.dot(v12, u12) ) * u12
    F_viscelast_13 =  - ( k2 * (d13 - L)  + c * np.dot(v13, u13) ) * u13
    F_viscelast_23 =  - ( k3 * (d23 - L)  + c * np.dot(v23, u23) ) * u23

    a1 =  g + (F_viscelast_12 + F_viscelast_13) / m1 
    a2 =  g + (F_viscelast_23 - F_viscelast_12) / m2
    a3 =  g - (F_viscelast_13 + F_viscelast_23) / m3

    return np.concatenate((v1,a1,v2,a2,v3,a3))

# --- Funciones auxiliares (plots y animaciones) ---

# Función para dar formato a los ejes
def formato_lanzamiento(ax):
    ax.set_xlabel(r'Alcance $x$ [m]')
    ax.set_ylabel(r'Altura $y$ [m]')
    # Detalles finos (Grilla y Ticks)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.8) # Eje cero
    ax.minorticks_on() # Ticks menores son esenciales para lectura precisa

    ax.minorticks_on() # Ticks menores para lectura precisa
    ax.legend(frameon=True, loc='best', fontsize=12, framealpha=0.6, edgecolor='gray')

def update_magnus_loop(frame):
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


def update2(frame):
    line1.set_data(x1[:frame], y1[:frame])
    line1.set_3d_properties(z1[:frame])
    
    line2.set_data(x2[:frame], y2[:frame])
    line2.set_3d_properties(z2[:frame])
    
    # Actualizar posiciones actuales
    point1.set_data([x1[frame]], [y1[frame]]) 
    point1.set_3d_properties([z1[frame]])
    
    point2.set_data([x2[frame]], [y2[frame]])
    point2.set_3d_properties([z2[frame]])
    
    # Actualizar muelle 
    muelle.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
    muelle.set_3d_properties([z1[frame], z2[frame]])
    
    return line1, point1, line2, point2, muelle



def update3(frame):
    line1.set_data(x1[:frame], y1[:frame])
    line1.set_3d_properties(z1[:frame])
    
    line2.set_data(x2[:frame], y2[:frame])
    line2.set_3d_properties(z2[:frame])
    
    line3.set_data(x3[:frame], y3[:frame])
    line3.set_3d_properties(z3[:frame])
    
    # Actualizar posiciones actuales
    point1.set_data([x1[frame]], [y1[frame]]) 
    point1.set_3d_properties([z1[frame]])
    
    point2.set_data([x2[frame]], [y2[frame]])
    point2.set_3d_properties([z2[frame]])
    
    point3.set_data([x3[frame]], [y3[frame]])
    point3.set_3d_properties([z3[frame]])
    
    # Actualizar muelles
    muelle12.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
    muelle12.set_3d_properties([z1[frame], z2[frame]])
    
    muelle13.set_data([x1[frame], x3[frame]], [y1[frame], y3[frame]])
    muelle13.set_3d_properties([z1[frame], z3[frame]])

    muelle23.set_data([x2[frame], x3[frame]], [y2[frame], y3[frame]])
    muelle23.set_3d_properties([z2[frame], z3[frame]])

    return line1, point1, line2, point2, line3, point3, muelle12, muelle13, muelle23


def update_practice3(frame):
    # El índice real en el array de datos
    idx = frame * step
    if idx >= len(x_vals):
        idx = len(x_vals) - 1
        
    line.set_data(x_vals[:idx], y_vals[:idx])
    line.set_3d_properties(z_vals[:idx])
    
    point.set_data([x_vals[idx]], [y_vals[idx]]) 
    point.set_3d_properties([z_vals[idx]])

    shadow.set_data(x_vals[:idx], y_vals[:idx])
    shadow.set_3d_properties(np.zeros(idx))


    idx_hook = frame * step
    if idx_hook >= len(x_hook):
        idx_hook = len(x_hook)-1
    
    line_h.set_data(x_hook[:idx_hook], y_hook[:idx_hook])
    line_h.set_3d_properties(z_hook[:idx_hook])
    
    point_h.set_data([x_hook[idx_hook]], [y_hook[idx_hook]]) 
    point_h.set_3d_properties([z_hook[idx_hook]])

    shadow_h.set_data(x_hook[:idx_hook], y_hook[:idx_hook])
    shadow_h.set_3d_properties(np.zeros(idx_hook))

    
    return line, point, shadow, line_h, point_h, shadow_h

# %% === Variación de la trayectoria en función del ángulo: Casos no aire y solo rozamiento ===

ang_range = np.arange(25, 80, 5) # Diferentes ángulos que vamos a emplear
diff_g = [] # Iniciailizamos vector diferencias altura g=cte vs g no cte
diff_a = [] # Ídem, para altura con y sin aire 

# Lo mismo para el alcance
delta_g = []
delta_a = []


fig = plt.figure(figsize=(24,10), dpi=120)

gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[0, 1])

ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])


for angulo in ang_range:
    # Condiciones iniciales.

    theta = np.radians(angulo)  #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    inicio = [x0,y0,vx0,vy0]

    # Caso ideal
    sol_no_air =  ja.euler_solver(caso_no_air, t, inicio, 1)
    x_na, y_na, vx_na, vy_na = sol_no_air[:, 0], sol_no_air[:,1], sol_no_air[:,2], sol_no_air[:,3]
    t_sol_no_air = t[:len(y_na)] # Truncamos el timpo para que tenga la misma dimensión que y
    ax1.plot(x_na, y_na, '--', alpha=0.5, label=rf'$\theta={angulo:.0f}^\circ$', zorder=1)

    # Caso gravedad variable (sin aire)
    sol_no_air_g =  ja.euler_solver(caso_no_air_g, t, inicio, 1)
    x_na_g, y_na_g, vx_na_g, vy_na_g = sol_no_air_g[:, 0], sol_no_air_g[:,1], sol_no_air_g[:,2], sol_no_air_g[:,3]
    t_sol_no_air_g = t[:len(y_na_g)] # Truncamos el timpo para que tenga la misma dimensión que y
    ax3.plot(x_na_g, y_na_g, '--', alpha=0.5, label=rf'$\theta={angulo:.0f}^\circ$', zorder=1)

    # A partir de aquí todo tiene gravedad variable

    # Caso solo rozamiento
    sol =  ja.euler_solver(caso_only_roz, t, inicio, 1)
    x, y, vx, vy = sol[:, 0], sol[:,1], sol[:,2], sol[:,3]
    t_sol = t[:len(y)] # Truncamos el timpo para que tenga la misma dimensión que y
    ax2.plot(x, y, '--', alpha=0.5, label=rf'$\theta={angulo:.0f}^\circ$', zorder=1)

    # Diferencia altura y recorrido Con y Sin Gravedad Variable
    diff_g.append(max(y_na_g) - max(y_na))
    delta_g.append(max(x_na_g) - max(x_na))

    # Diferencia altura y recorrido Con y Sin Aire
    diff_a.append(max(y) - max(y_na))
    delta_a.append(max(x) - max(x_na))
    

# Con y Sin Gravedad Variable
ax4.plot(ang_range, diff_g, ':', alpha=0.8, label=rf'Altura máxima', zorder=1)
ax4.plot(ang_range, delta_g, ':', alpha=0.8, label=rf'Alcance máximo', zorder=1)

# Con y Sin Aire
ax5.plot(ang_range, diff_a, ':', alpha=0.8, label='Altura máxima', zorder= 1)
ax5.plot(ang_range, delta_a, ':', alpha=0.8, label=rf'Alcance máximo', zorder=1)


# Formato de Ejes y Etiquetas

formato_lanzamiento(ax1)
formato_lanzamiento(ax2)
formato_lanzamiento(ax3)

ax1.set_title(r'Trayectoria Proyectil - Sin Aire + Gravedad Constante')
ax2.set_title(r'Trayectoria Proyectil - Con Aire + Gravedad Variable')
ax3.set_title(r'Trayectoria Proyectil - Sin Aire + Gravedad Variable')

ax4.set_xlabel(rf"Ángulo $\theta$ [$^\circ$]")
ax4.set_ylabel("Diferencia de distancia [m]")
ax4.set_title(rf"$\Delta$ distancias máximas: Gravedad Constante vs Variable")
ax4.legend()

ax5.set_xlabel(rf"Ángulo $\theta$ [$^\circ$]")
ax5.set_ylabel("Diferencia de distancia [m]")
ax5.set_title(rf"$\Delta$ distancias máximas: Con Aire vs Sin Aire")
ax5.legend()

# Guardado y Muestra
plt.tight_layout()
plt.savefig(figuras/'ideal_vs_g_vs_air.png', bbox_inches='tight', dpi = 200) # Formato vectorial preferido
plt.show()
# plt.close()

print('='*60)
print('La diferencia es claramente notable en el caso Con y Sin Aire.')
print('Para el caso de gravedad cte vs variable, no es tan relevante.\nSin embargo, consideraremos a partir de ahora gravedad variable.')
print('='*60)


# %% === Comparación Distintos Modelos (Mismo Ángulo)

# Condiciones inciales
theta = np.radians(45) #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
inicio = [x0,y0, v0 * np.cos(theta), v0 * np.sin(theta)]

# Caso ideal
sol_no_air = ja.euler_solver(caso_no_air, t, inicio, 1)
no_air = sol_no_air.T
t_no_air = t[:len(no_air[1])] # Truncamos el timpo para que tenga la misma dimensión que y

# Caso solo rozamiento
sol_only_roz = ja.euler_solver(caso_only_roz, t, inicio, 1)
only_roz = sol_only_roz.T
t_only_roz = t[:len(only_roz[1])] # Truncamos el timpo para que tenga la misma dimensión que y


# A partir de aquí tiene viento en la dirección x. 

# Modelo isotérmico
sol_isot =  ja.euler_solver(caso_isotermo, t, inicio, 1)
isot = sol_isot.T
t_isot = t[:len(isot[1])] # Truncamos el timpo para que tenga la misma dimensión que y

# Modelo Adiabático
sol_adiab =  ja.euler_solver(caso_adiab, t, inicio, 1)
adiab = sol_adiab.T
t_adiab = t[:len(adiab[1])] # Truncamos el timpo para que tenga la misma dimensión que y

# Efecto Magnus. Ojo, estamos considerando w en el eje z, que NO es dirección de movimiento (el típico eje z es el eje y)
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
ax.set_title(rf'Trayectoria Proyectil - Ángulo $\theta = {np.degrees(theta):.0f}^\circ$')

# Guardado y Muestra
plt.tight_layout()
plt.savefig(figuras/'all_models_45.png', bbox_inches='tight', dpi=200)
plt.show()
# plt.close()

# %% === Efecto Magnus === 

theta = np.radians(30)  #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
vx0 = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)

inicio = [x0,y0,vx0,vy0]

sol_magnus = ja.euler_solver(caso_magnus, t, inicio, 1)
magnus = sol_magnus.T
t_magnus = t[:len(magnus[1])] # Truncamos el timpo para que tenga la misma dimensión que y


fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

ax.plot(magnus[0], magnus[1], '--', color = 'fuchsia', label = 'Magnus', alpha=0.5, zorder=1)

formato_lanzamiento(ax)
ax.legend().remove()
ax.set_title(rf'Loop por Efecto Magnus $-$ Ángulo $\theta = {np.degrees(theta):.0f}^\circ$')

# Guardado y Muestra
plt.tight_layout()
plt.savefig(figuras/'loop_magnus_2D.png', bbox_inches='tight', dpi=200)
plt.show()
# plt.close()


# %% === Magnus en 3D (Sin Animar) ===

theta = np.radians(30)  #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
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

ax.set_zlim(0, np.max(mag_3D[1]))

# Para limitar el número de números que aparecen en los ejes
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
ax.zaxis.set_major_locator(MaxNLocator(nbins=5))

ax.set_xlabel('$x$ [m]', labelpad = 15)
ax.set_ylabel('$z$ [m]', labelpad = 15)
ax.set_zlabel('$y$ [m]', labelpad = 15)

ax.set_box_aspect((2, 1, 2)) # Escojo un tamaño de box particular (eje z más estrecho)

ax.set_title(rf'Loop por Efecto Magnus $-$ Ángulo $\theta = {np.degrees(theta):.0f}^\circ$')

plt.savefig(figuras/'loop_magnus_3D.png', bbox_inches='tight', dpi=200)
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
ax.set_ylim(-0.5,0.5)
ax.set_zlim(0, np.max(y_vals))


# Para limitar el número de números que aparecen en los ejes
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
ax.zaxis.set_major_locator(MaxNLocator(nbins=5))


ax.set_box_aspect((2, 1, 2)) # Escojo un tamaño de box particular (eje z más estrecho)
ax.set_title(rf'Loop por Efecto Magnus $-$ Ángulo $\theta = {np.degrees(theta):.0f}^\circ$')


frames_totales = len(x_vals)//20
step = max(1, len(x_vals) // frames_totales)


ani = animation.FuncAnimation(
    fig, 
    update_magnus_loop, 
    frames=frames_totales, 
    interval=1, # milisegundos entre frames
    blit=False,   # blit=False suele ser más estable en 3D
    repeat = False
)

# Para guardar la animación. 

# ani.save(
#     figuras/"ani_loop_magnus.mp4",
#     writer="ffmpeg",
#     fps=30,
#     dpi=200
# )

plt.show()

# %% === Optimización no inteligente: Buscamos el mejor (mayor alcance) de todos los posibles ángulos ===

print('='*50)
print('-'*50)
print('Este proceso puede tardar un poco.')



angle = np.arange(0,50,0.1)

N = int(1e5)
t = np.linspace(0,1000,N)

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

    if ang % 10 == 0:
        print(f"Optimización en proceso. Ángulo actual: {ang}")

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
        f.write(f"{modelo:<10}\t{ang:>10.1f}\t{alcance:>10.1f}\n")
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

ax.plot(no_air[0], no_air[1], '--', alpha=0.5, label=rf'Sin Aire $\theta = {np.degrees(best_no_air[0]):.1f}^\circ$', zorder=1)
ax.plot(only_roz[0], only_roz[1], '--', alpha=0.5, label=rf'Con aire $\theta = {np.degrees(best_only_roz[0]):.1f}^\circ$', zorder=1)
ax.plot(isot[0], isot[1], '--', alpha=0.5, label=rf'Isotérmico $\theta = {np.degrees(best_isot[0]):.1f}^\circ$', zorder=1)
ax.plot(adiab[0], adiab[1], '--', alpha=0.5, label=rf'Adiabático $\theta = {np.degrees(best_adiab[0]):.1f}^\circ$', zorder=1)
ax.plot(magnus[0], magnus[1], '--', alpha=0.5, label=rf'Magnus $\theta = {np.degrees(best_magnus[0]):.1f}^\circ$', zorder=1)

formato_lanzamiento(ax)
ax.set_title(r'Trayectoria Proyectil - Ángulo Lanzamiento Óptimo')

plt.tight_layout()
plt.savefig(figuras/'all_models_optim.png', bbox_inches='tight', dpi=200)
plt.show()
# plt.close()



# %% === LANZANDO PELOTAS CON MUELLES ===

# --- Valores Físicos ---

m1, m2, m3 = 1, 2, 3 # [kg] Masas de los cuerpos
L = 1.0 # [m] Longitud de la muelle (todos los muelles iguales)
k = 100.0 # [N/m] constante elástica muelle
k2 = k/50.0 # [N/m] constante elástica muelle 2
k3 = k*50 # [N/m] constante elástica muelle 3
c = 30.0 # [N * s / m] constante viscosa del fluido
g = np.array([0,0, -sp.constants.g]) #Constante gravitatoria en formato vector (vamos a trabajar con vectores)

t = np.linspace(0,10, 10000) # Volvemos a declarar un nuevo tiempo para no utilizar tanto como empleamos en los anteriores lanzamientos


# --- 2 PELOTAS ---
y0 = [-10, 10, 0,     # r1
      0, 0, 10,       # v1
      10, 0, 0,       # r2
      -20, -10, 20]   # v2


sol_visc_elas_all = ja.rk4_solver(muelle_viscoelástico2, t, y0, 2)
sol_visc_elas = sol_visc_elas_all.T

R1 = sol_visc_elas[0:3]
V1 = sol_visc_elas[3:6]
R2 = sol_visc_elas[6:9]
V2 = sol_visc_elas[9:12]

x1, y1, z1 = R1[0:3]
x2, y2, z2 = R2[0:3]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

# Elementos gráficos a actualizar
line1, = ax.plot([], [], [], 'b--', lw=1, alpha=0.5, label='Trayectoria Cuerpo 1')
point1, = ax.plot([], [], [], 'bo', ms=8)
line2, = ax.plot([], [], [], 'r--', lw=1, alpha=0.5, label='Trayectoria Cuerpo 2')
point2, = ax.plot([], [], [], 'ro', ms=8)
muelle, = ax.plot([], [], [], 'k-', lw=2)

# Configuración de ejes FIJA 
ax.set_xlim(min(np.min(x1), np.min(x2)), max(np.max(x1), np.max(x2)))
ax.set_ylim(min(np.min(y1), np.min(y2)), max(np.max(y1), np.max(y2)))
ax.set_zlim(0, max(np.max(z1), np.max(z2)) + 1)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('2 Cuerpos unidos por muelle viscoelástico')
ax.legend()

# Crear animación
ani = FuncAnimation(
    fig, 
    update2,
    frames=range(0,len(t), 20), # Con esto puedo modificar rapidez del vídeo
    interval=2,
    blit=False
    )

# Para guardar la animación. 

# ani.save(
#     figuras/"lanzamiento_2.mp4",
#     writer="ffmpeg",
#     fps=30,
#     dpi=200
# )

plt.show()



#--- 3 cuerpos ---

t = np.linspace(0,20, 10000)

y0 = [-1, 1, 1,   # r1
      2, 0, 5,    # v1
      0, 2, 2,    # r2
     -2, 0, 2,   # v2
     0, 10, 0,    # r3
      2, 0, 2     # v3
    ]


sol_visc_elas_all = ja.rk4_solver(muelle_viscoelástico3, t, y0)

sol_visc_elas = sol_visc_elas_all.T

R1 = sol_visc_elas[0:3]
V1 = sol_visc_elas[3:6]
R2 = sol_visc_elas[6:9]
V2 = sol_visc_elas[9:12]
R3 = sol_visc_elas[12:15]
V3 = sol_visc_elas[15:18]


x1, y1, z1 = R1[0:3]
x2, y2, z2 = R2[0:3]
x3, y3, z3 = R3[0:3]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')


# Elementos gráficos a actualizar
line1, = ax.plot([], [], [], 'b--', lw=1, alpha=0.5, label='Trayectoria Cuerpo 1')
point1, = ax.plot([], [], [], 'bo', ms=8)
line2, = ax.plot([], [], [], 'r--', lw=1, alpha=0.5, label='Trayectoria Cuerpo 2')
point2, = ax.plot([], [], [], 'ro', ms=8)
line3, = ax.plot([], [], [], 'g--', lw=1, alpha=0.5, label='Trayectoria Cuerpo 3')
point3, = ax.plot([], [], [], 'go', ms=8)
muelle12, = ax.plot([], [], [], 'k-', lw=2) 
muelle13, = ax.plot([], [], [], 'k-', lw=2) 
muelle23, = ax.plot([], [], [], 'k-', lw=2) 

# Configuración de ejes FIJA 
ax.set_xlim(min(np.min(x1), np.min(x2), np.min(x3)), max(np.max(x1), np.max(x2), np.max(x3)))
ax.set_ylim(min(np.min(y1), np.min(y2), np.min(y3)), max(np.max(y1), np.max(y2), np.max(y3)))
ax.set_zlim(min(np.min(z1), np.min(z2), np.min(z3)), max(np.max(z1), np.max(z2), np.max(z3)) + 1)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3 Cuerpos unidos por muelle viscoelástico')
ax.legend()


# Crear animación
ani = FuncAnimation(fig, update3, frames=range(0,len(t), 20), # Con esto puedo controlar rapidez animación
                     interval=10, blit=False)

# ani.save(
#     figuras/"lanzamiento_3.mp4",
#     writer="ffmpeg",
#     fps=30,
#     dpi=200
# )

plt.show()


# %% === SLICE y HOOK (PARA PELOTA DE GOLF) ===

# --- PARÁMETROS ---
v0 = 70 # [m/s] Velocidad inicial en módulo. 
p0 = 1.225 # [kg/m**3] Densidad del aire a nivel del mar
g = sp.constants.g
RT = 6371000 #[m] Radio terrestre
a = 6.5e-3 #Valor de a en la fórmula adiabática
alpha = 2.5 # Exponente en la fórmula adibática
T0 = 293 #[K] Temperatura ambiente, para adibática
S0wm = 0.25 # [1/m]
R_golf = 0.02 # [m] Radio pelota de golf
A_golf = np.pi * R_golf**2
m_golf = 0.045 #[kg] Masa pelota de golf

# --- Valores iniciales e intervalo de tiempo ---

x0, y0, z0 = 0, 0, 0
vwx = 10 # [m/s] Velocidad del viento en la dirección x. 

t = np.linspace(0,10,10000)


def magnus_3D_slice(t, vec_ini):

    x, y, z = vec_ini[0], vec_ini[1], vec_ini[2]
    vx, vy, vz = vec_ini[3], vec_ini[4], vec_ini[5]

#   Velocidades
    dx = vx 
    dx_rel = dx - vwx #velocidad real (quitandole la componente del viento)
    dy = vy
    dz = vz
    
    v = np.sqrt(dx_rel**2 + dy**2 + dz**2) 

    if v < 14:
        C = 1.0
    else:
        C = 14 / v
    
    rho = (1 - (a * z / T0)) ** alpha #En unidades de rho_0, de tal manera que no haya que dividir después por rho_0

    arrastre = C * rho * A_golf / m_golf

    dvx = - arrastre * v * dx_rel - S0wm * dy
    dvy = - arrastre * v * dy     + S0wm * dx_rel
    dvz = -g *(RT / (RT + z)) ** 2 - arrastre * v * dz 

    return np.array([dx, dy, dz, dvx, dvy, dvz])

def magnus_3D_hook(t, vec_ini):
    x, y, z = vec_ini[0], vec_ini[1], vec_ini[2]
    vx, vy, vz = vec_ini[3], vec_ini[4], vec_ini[5]

#   Velocidades
    dx = vx 
    dx_rel = dx - vwx #velocidad real (quitandole la componente del viento)
    dy = vy
    dz = vz
    
    v = np.sqrt(dx_rel**2 + dy**2 + dz**2) 

    if v < 14:
        C = 1.0
    else:
        C = 14 / v
    
    rho = (1 - (a * z / T0)) ** alpha

    arrastre = C * rho * A_golf / m_golf

    dvx = -arrastre * v * dx_rel + S0wm * dy
    dvy = -arrastre * v * dy     - S0wm * dx_rel
    dvz = -g *(RT / (RT + z)) ** 2 - arrastre * v * dz 

    return np.array([dx, dy, dz, dvx, dvy, dvz])


# Buscamos el valor de ángulo que maximiza el alcance en x.

x_max = 0
ang_optim = 0

for i, ang in enumerate(np.arange(0,50,0.1)):
    theta = np.radians(ang)  #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
    vx0 = v0 * np.cos(theta)
    vy0 = 0
    vz0 = v0 * np.sin(theta)

    inicial_state = [x0,y0,z0,vx0,vy0,vz0]

    # Solo necesitamos evaluar 1 de los dos posibles tiros (hook o slice) porque son simétricos, ya que 
    # solamente estamos considerando velocidad del viento en la dirección x.
    sol_magnus_3D = ja.euler_solver(magnus_3D_slice, t, inicial_state, 2)
    mag_3D = sol_magnus_3D.T

    x_i_max = np.max(mag_3D[0])

    if x_i_max > x_max:
        x_max = x_i_max
        ang_optim = ang

# --- Graficamos el lanzamiento con mayor alcance
theta = np.radians(ang_optim)  #Como tenemos el módulo de la velocidad, debemos poner un ángulo de lanzamiento 
vx0 = v0 * np.cos(theta)
vy0 = 0
vz0 = v0 * np.sin(theta)

inicial_state = [x0,y0,z0,vx0,vy0,vz0]

sol_magnus_3D = ja.euler_solver(magnus_3D_slice, t, inicial_state, 2)
mag_3D = sol_magnus_3D.T
t_magnus_3D = t[:len(mag_3D[2])]

sol_magnus_3D_neg = ja.euler_solver(magnus_3D_hook, t, inicial_state, 2)
mag_3D_neg = sol_magnus_3D_neg.T
t_magnus_3D_neg = t[:len(mag_3D_neg[2])]



import matplotlib.animation as animation

fig = plt.figure(figsize=(10, 8), dpi=100)
ax = fig.add_subplot(projection='3d')

line, = ax.plot([], [], [], lw=2, color='violet', label='Slice') 
point, = ax.plot([], [], [], 'o', color='crimson', markersize=5) # El punto que está marcando el movimiento
shadow, = ax.plot([], [], [], '--', color='gray', alpha=0.5) # Proyección en el plano del suelo


line_h, = ax.plot([], [], [], lw=2, color='green', label='Hook') 
point_h, = ax.plot([], [], [], 'o', color='lime', markersize=5) # El punto que está marcando el movimiento
shadow_h, = ax.plot([], [], [], '--', color='gray', alpha=0.5) # Proyección en el plano del suelo


ax.set_xlabel('$x$ [m]', labelpad = 15)
ax.set_ylabel('$y$ [m]', labelpad = 15)
ax.set_zlabel('$z$ [m]', labelpad = 15)


x_vals = mag_3D[0]
y_vals = mag_3D[1]
z_vals = mag_3D[2]

r_vals_mod = np.sqrt(x_vals**2 + y_vals**2 + z_vals**2)

vx_vals = mag_3D[3]
vy_vals = mag_3D[4]
vz_vals = mag_3D[5]

v_vals_mod  = np.sqrt(vx_vals**2 + vy_vals**2 + vz_vals**2)

x_hook = mag_3D_neg[0]
y_hook = mag_3D_neg[1]
z_hook = mag_3D_neg[2]

r_hook_mod = np.sqrt(x_hook**2 + y_hook **2 + z_hook**2)

vx_hook = mag_3D_neg[3]
vy_hook = mag_3D_neg[4]
vz_hook = mag_3D_neg[5]

v_hook_mod = np.sqrt(vx_hook**2 + vy_hook**2 + vz_hook**2)

ax.set_xlim(min(np.min(x_vals), np.min(x_hook)), max(np.max(x_vals), np.max(x_hook)))
ax.set_ylim(min(np.min(y_vals), np.min(y_hook)), max(np.max(y_vals), np.max(y_hook)))
ax.set_zlim(0, max(np.max(z_vals), np.max(z_hook)))
ax.set_title('Hook y Slice')
ax.legend()

# Para limitar el número de números que aparecen en los ejes
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax.zaxis.set_major_locator(MaxNLocator(nbins=5))

frames_totales =  max(len(x_vals), len(x_hook))//100
step = max(1, max(len(x_vals), len(x_hook))//frames_totales)


ani = animation.FuncAnimation(
    fig, 
    update_practice3, 
    frames=frames_totales, 
    interval=1,  # milisegundos entre frames
    blit=False   # blit=False suele ser más estable en 3D
)

# ani.save(
#     figuras/"hook_y_slice_fast.mp4",
#     writer="ffmpeg",
#     fps=30,
#     dpi=200
# )

# fig.savefig(figuras/"hook_y_slice.png", bbox_inches='tight', dpi = 200)

plt.show()

# --- Resultados y Espacio de Fases

print('='*75)
print(f"Ángulo que maximiza distancia recorrida: {ang_optim:.1f}º")
print('='*75)


fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

# Graficado
# Usar zorder para controlar superposición
ax.plot(v_vals_mod, r_vals_mod, ':', color='magenta', alpha = 0.5, label=r'Slice', zorder=1)
ax.plot(v_hook_mod, r_hook_mod, '-.', color='green', alpha=0.5, label=r'Hook', zorder=1)

# Formato de Ejes y Etiquetas
ax.set_xlabel(r'Velocidad $v$ [m/s]')
ax.set_ylabel(r'Posición $r$ [m]')
ax.set_title(r'Espacio de Fases')

ax.minorticks_on()

# Leyenda (para eso sirve label)
ax.legend(frameon=True, loc='best', fontsize=12, framealpha=0.9, edgecolor='gray')

# Guardado y Muestra
plt.tight_layout()
plt.savefig(figuras/'espacio_fases_golf.png', bbox_inches='tight', dpi = 200) 
plt.show()


plt.close()
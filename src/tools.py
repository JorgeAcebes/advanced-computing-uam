import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
import scipy as sp
# __________________________________________

#                IVP SOLVERS
# __________________________________________

# ==========================================
# 1. EULER EXPLÍCITO (Orden 1)
# ==========================================
def euler_solver(func, t_array, y0, stop_idx=None):
    """
    Resuelve dy/dt = func(t, y) usando Euler Explícito.
    Orden global: O(dt).

    NO CONSERVA ENERGÍA.

    Args:
    func: Función callable f(t, y) -> array_like (derivadas)
    t_array: Array de tiempos
    y0: Condiciones iniciales (array)
    stop_idx (int, optional): Índice de la variable a monitorear. 
    =============================================================

    Ejemplo de uso:

    # --- VALORES INICIALES ---
    omega = 2.0
    x0_val = 1.0  # Posición inicial
    v0_val = 0.0  # Velocidad inicial
    t = np.linspace(0, 20, 1000) # 20 segundos

    # --- DEFINICIÓN DE SISTEMAS ---

    # Estado y = [x, v]
    def oscilador_derivada(t, y):
        x, v = y
        dxdt = v
        dvdt = -(omega**2) * x
        return np.array([dxdt, dvdt])
    
    # --- EJECUCIÓN ---

    sol_euler = euler_solver(oscilador_derivada, t, y0)
    x_euler = sol_euler[:, 0]   

    Desempaquetar todo los datos mas eficientemente:

    x,v  = sol_euler.T 

    """
    steps = len(t_array)
    dt = t_array[1] - t_array[0]
    
    y_sol = np.zeros((steps, len(y0)))
    y_sol[0] = y0
    
    for i in range(steps - 1):
        y_next = y_sol[i] + dt * func(t_array[i], y_sol[i])
        
        if stop_idx is not None and y_next[stop_idx] < 0:
            
            #  Lógica de Interpolación (Aterrizaje suave) 
            # y_prev es positivo, y_next es negativo.
            y_prev = y_sol[i, stop_idx]
            val_next = y_next[stop_idx]
            
            # Fracción del paso dt necesaria para llegar exactamente a 0
            frac = y_prev / (y_prev - val_next)
            
            # Calculamos el estado final interpolado
            y_sol[i+1] = y_sol[i] + frac * (y_next - y_sol[i])
            
            # Devolvemos el array CORTADO hasta el impacto (i+2)
            return y_sol[:i+2]

        # 3. Si no hay parada o no se cumplió, seguimos normal
        y_sol[i+1] = y_next
        
    return y_sol


# ==========================================
# 1.1 EULER - CROMER (Orden 1)
# ==========================================
def euler_cromer_solver(func, t_array, y0, stop_idx=None):
    """
    Resuelve dy/dt = func(t, y) usando Euler Cromer.


    Args:
    func: Función callable f(t, y) -> array_like (derivadas)
    t_array: Array de tiempos
    y0: Condiciones iniciales (array)
    stop_idx (int, optional): Índice de la variable a monitorear. 
    =============================================================

    Ejemplo de uso:

    # --- VALORES INICIALES ---
    omega = 2.0
    x0_val = 1.0  # Posición inicial
    v0_val = 0.0  # Velocidad inicial
    t = np.linspace(0, 20, 1000) # 20 segundos

    # --- DEFINICIÓN DE SISTEMAS ---

    # Estado y = [x, v]
    def oscilador_derivada(t, y):
        x, v = y
        dxdt = v
        dvdt = -(omega**2) * x
        return np.array([dxdt, dvdt])
    
    # --- EJECUCIÓN ---

    sol_euler_cromer = euler_cromer_solver(oscilador_derivada, t, y0)
    x_euler = sol_euler_cromer[:, 0]   

    Desempaquetar todo los datos mas eficientemente:

    x,v  = sol_euler_cromer.T 

    """
    steps = len(t_array)
    dt = t_array[1] - t_array[0]
    dim = len(y0)
    half_dim = dim // 2  # Punto de corte entre posición y velocidad
    
    y_sol = np.zeros((steps, dim))
    y_sol[0] = y0
    
    for i in range(steps - 1):
        t_curr = t_array[i]
        y_curr = y_sol[i]
        
        # 1. Obtenemos derivadas [v_n, a_n] usando el estado actual
        derivs = func(t_curr, y_curr)
        
        # Inicializamos el siguiente paso
        y_next = np.empty_like(y_curr)
        
        # 2. Actualizamos VELOCIDAD primero (v_{n+1} = v_n + a_n * dt)
        # Usamos la segunda mitad de 'derivs' (aceleraciones)
        y_next[half_dim:] = y_curr[half_dim:] + derivs[half_dim:] * dt
        
        # 3. Actualizamos POSICIÓN usando la NUEVA velocidad (x_{n+1} = x_n + v_{n+1} * dt)
        # Ignoramos la primera mitad de 'derivs' (que sería v_n)
        y_next[:half_dim] = y_curr[:half_dim] + y_next[half_dim:] * dt
        
        # --- Lógica de evento (Stop condition) ---
        if stop_idx is not None and y_next[stop_idx] < 0:
            y_prev = y_curr[stop_idx]
            val_next = y_next[stop_idx]
            frac = y_prev / (y_prev - val_next)
            
            # Interpolación lineal para todo el vector
            y_sol[i+1] = y_curr + frac * (y_next - y_curr)
            return y_sol[:i+2]

        y_sol[i+1] = y_next
        
    return y_sol

# ==========================================
# 2. RUNGE-KUTTA 4 (Orden 4)
# ==========================================
def rk4_solver(func, t_array, y0, stop_idx=None):
    """
    Resuelve dy/dt = func(t, y) usando RK4 clásico.
    Orden global: O(dt^4).

    dy/dt = func(t, y)
    
    Args:
        func: Función callable f(t, y) -> array_like (derivadas)
        t_array: Array de tiempos
        y0: Condiciones iniciales (array)
        stop_idx (int, optional): Índice de la variable a monitorear para parada.
        
    Returns:
        y: Matriz de soluciones de forma (len(t), len(y0))
    
        
    =============================================================

    Ejemplo de uso:

    # --- VALORES INICIALES ---
    omega = 2.0
    x0_val = 1.0  # Posición inicial
    v0_val = 0.0  # Velocidad inicial
    t = np.linspace(0, 20, 1000) # 20 segundos

    # --- DEFINICIÓN DE SISTEMAS ---

    # Estado y = [x, v]
    def oscilador_derivada(t, y):
        x, v = y
        dxdt = v
        dvdt = -(omega**2) * x
        return np.array([dxdt, dvdt])
    
    # --- EJECUCIÓN ---
    
    y0 = np.array([x0_val, v0_val])
    sol_rk4 = rk4_solver(oscilador_derivada, t, y0)
    x_rk4 = sol_rk4[:, 0] # Extraemos posición


    Desempaquetar todo los datos mas eficientemente:

    x,v  = sol_rk4.T
    """
    steps = len(t_array)
    dt = t_array[1] - t_array[0]
    
    y_sol = np.zeros((steps, *np.shape(y0)))
    y_sol[0] = y0
    
    for i in range(steps - 1):
        t_i = t_array[i]
        y_i = y_sol[i]
        
        k1 = func(t_i, y_i)
        k2 = func(t_i + 0.5*dt, y_i + 0.5*dt*k1)
        k3 = func(t_i + 0.5*dt, y_i + 0.5*dt*k2)
        k4 = func(t_i + dt, y_i + dt*k3)
        
        # Calculamos el siguiente paso candidato
        y_next = y_i + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Lógica de parada
        if stop_idx is not None and y_next[stop_idx] < 0:
            y_prev_val = y_i[stop_idx]
            y_next_val = y_next[stop_idx]
            
            # Interpolación lineal
            frac = y_prev_val / (y_prev_val - y_next_val)
            y_sol[i+1] = y_i + frac * (y_next - y_i)
            
            return y_sol[:i+2]

        y_sol[i+1] = y_next
        
    return y_sol

# ==========================================
# 3. VELOCITY VERLET (Orden 2)
# ==========================================
def verlet_solver(acc_func, t_array, x0, v0, stop_idx=None):
    """
    Resuelve d^2x/dt^2 = a(x).
    Orden global: O(dt^2). Preserva energía en sistemas conservativos.
    
    Args:
        acc_func: Función a(x). NO depende de v ni t.
        x0, v0: Condiciones iniciales (pueden ser vectores N-dimensionales).
        stop_idx (int, optional): Índice de la coordenada de posición 'x' a monitorear.
    Returns:
        x_sol, v_sol


    =======================================================================

    Ejemplo de uso:

    # --- VALORES INICIALES ---
    omega = 2.0
    x0_val = 1.0  # Posición inicial
    v0_val = 0.0  # Velocidad inicial
    t = np.linspace(0, 20, 1000) # 20 segundos

    # --- DEFINICIÓN DE SISTEMAS ---
    def oscilador_aceleracion(x):
    return -(omega**2) * x
    
    # --- EJECUCIÓN ---
    # Nota: Pasamos x0 y v0 por separado, y la función de aceleración solo depende de x
    x_verlet, v_verlet = verlet_solver(oscilador_aceleracion, t, x0_val, v0_val)

    """
    steps = len(t_array)
    dt = t_array[1] - t_array[0]
    
    # Pre-allocating arrays con la forma de x0 (ej: (steps, N_particulas, 3))
    x_sol = np.zeros((steps, *np.shape(x0)))
    v_sol = np.zeros((steps, *np.shape(v0)))
    
    x_sol[0] = x0
    v_sol[0] = v0
    
    # Pre-cálculo de la aceleración inicial
    a_current = acc_func(x0)
    
    for i in range(steps - 1):
        # 1. Kick (v a medio paso)
        v_half = v_sol[i] + 0.5 * a_current * dt
        
        # 2. Drift (x paso completo)
        x_next = x_sol[i] + v_half * dt
        
        # 3. Calcular aceleración en nueva posición
        # (Necesaria para completar el paso de velocidad)
        a_next = acc_func(x_next)
        
        # 4. Kick (v paso completo)
        v_next = v_half + 0.5 * a_next * dt
        
        # Lógica de parada (Chequeamos x_next)
        if stop_idx is not None and x_next[stop_idx] < 0:
            x_prev_val = x_sol[i][stop_idx]
            x_next_val = x_next[stop_idx]
            
            # Interpolación lineal
            frac = x_prev_val / (x_prev_val - x_next_val)
            
            # Interpolamos tanto posición como velocidad
            x_sol[i+1] = x_sol[i] + frac * (x_next - x_sol[i])
            v_sol[i+1] = v_sol[i] + frac * (v_next - v_sol[i])
            
            return x_sol[:i+2], v_sol[:i+2]

        # Si no paramos, guardamos los valores calculados
        x_sol[i+1] = x_next
        v_sol[i+1] = v_next
        
        # Actualizar a_current para el siguiente ciclo (evita re-evaluación)
        a_current = a_next
        
    return x_sol, v_sol


# __________________________________________

#           AUXILIARY FUNCTIONS
# __________________________________________


# Angles and Chaos

def restringir(o):
    o = (o + np.pi) % (2*np.pi) - np.pi # Restrinjo el ángulo entre -pi y pi
    return o

def sec_poincare(o, w, t, tol = None):
    """
    Devuelve los elementos del np.array w tal que wt = 2*pi*n, así como los correspondientes ángulos o a los que sucede.

    o: np.array -- Ángulos
    w: np.array -- Velocidad angular 
    t: np.array -- Tiempos
    tol: int    -- Tolerancia de la igualdad
    """

    tol = (t[1]-t[0])/2 if tol is None else tol

    mask = ((w*t)%(2*np.pi)) < tol

    o_poinc = o[mask] 
    w_poinc = w[mask]
    t_poinc = t[mask]

    return o_poinc, w_poinc, t_poinc


def sec_poincare_forz(o, w, O_d, t, tol = None):
    """
    Devuelve los elementos del np.array w tal que wt = 2*pi*n, así como los correspondientes ángulos o a los que sucede.

    o: np.array -- Ángulos
    w: np.array -- Velocidad angular 
    t: np.array -- Tiempos
    tol: int    -- Tolerancia de la igualdad
    """

    tol = (t[1]-t[0])/2 if tol is None else tol

    mask = ((O_d*t)%(2*np.pi)) < tol

    o_poinc = o[mask] 
    w_poinc = w[mask]
    t_poinc = t[mask]

    return o_poinc, w_poinc, t_poinc


def amplitud(o, frac = 1/2):
    """
    Dado un vector de ángulos, devuelve la amplitud de la parte estacionaria

    == Input ==
    o: np.array -- Vector de ángulos
    frac: float -- Fracción del vector o que corresponde a la parte estacionaria. 
    Se considera por defecto que la segunda mitad del vector es la parte estacionaria.
    """
    if not 0 < frac <= 1:
        raise ValueError("frac debe estar en el intervalo (0, 1].")
    
    long_estacionaria = int(len(o) * frac)
    return  (np.max(o[long_estacionaria:]) - np.min(o[long_estacionaria:]))/2


# __________________________________________

#               PLOT STYLES
# __________________________________________

# Estilo texto y líneas ejes
def setup_style(base_size=13, dpi=150, **kwargs):
    """
    Configura el estilo global. 
    Usa kwargs para sobreescribir cualquier parámetro de rcParams.
    """
    config = {
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.serif": ["DejaVu Serif"],
        "figure.dpi": dpi,
        "xtick.labelsize": base_size,
        "ytick.labelsize": base_size,
        "axes.labelsize": base_size + 1,
        "axes.titlesize": base_size + 2,
        "legend.fontsize": base_size - 1,
        "figure.titlesize": base_size + 4
    }
    # Actualiza con parámetros específicos que pases (ej. setup_style(xtick_labelsize=20))
    config.update({k.replace('_', '.'): v for k, v in kwargs.items()})
    plt.rcParams.update(config)

def multiline_plot(ax, x, y_list, labels, colors, styles=None, **kwargs):
    """Grafica múltiples series de datos en un solo eje de forma compacta. Líneas continuas"""
    if styles is None: styles = ['-'] * len(y_list)
    for y, label, color, style in zip(y_list, labels, colors, styles):
        ax.plot(x, y, label=label, color=color, linestyle=style, **kwargs)

def multiscatter_plot(ax, x_list, y_list, labels, colors, **kwargs):
    """Grafica múltiples series de datos en un solo eje de forma compacta. Puntos (scatter)"""
    for x,y, label, color in zip(x_list, y_list, labels, colors):
        ax.scatter(x, y, label=label, color=color, **kwargs)


# Estilo etiquetas ejes
def setup_ax(ax, title=None, xlabel=None, ylabel=None, grid=False, legend=False, **kwargs):
    """Configuración estética. Solo aplica fuentes si se pasan por kwargs."""
    
    # Extraemos tamaños solo si existen, si no, dejamos que rcParams mande
    t_size = kwargs.get('titlesize')
    l_size = kwargs.get('labelsize')

    if title: 
        ax.set_title(title, fontsize=t_size) if t_size else ax.set_title(title)
    if xlabel: 
        ax.set_xlabel(xlabel, fontsize=l_size) if l_size else ax.set_xlabel(xlabel)
    if ylabel: 
        ax.set_ylabel(ylabel, fontsize=l_size) if l_size else ax.set_ylabel(ylabel)
    
    if grid: 
        ax.grid(True, linestyle=kwargs.get('grid_style', ':'), alpha=kwargs.get('alpha', 0.6))
    
    if legend: 
        ax.legend(frameon=True, loc='best', 
                  fontsize=kwargs.get('legendsize'), 
                  framealpha=kwargs.get('framealpha', 0.6))
        
          
# Estilo de ejes habituales
def espacio_tiempo(ax, espacio =r'Posición $x$ [m]'):
    ax.set_xlabel(r'Tiempo $t$ [s]')
    ax.set_ylabel(rf'{espacio}')
    ax.legend(frameon=True, loc='best', fontsize=12, framealpha=0.6, edgecolor='gray')

def espacio_fases(ax):
    ax.set_xlabel(r'Posición $x$ [m]')
    ax.set_ylabel(r'Velocidad $v$ [m/s]')
    ax.legend(frameon=True, loc='best', fontsize=12, framealpha=0.6, edgecolor='gray')



def animar_trayectorias(datos, duracion, fps=30, guardar=False, archivo='animacion.mp4', **kwargs):
    """
    datos: Lista de arrays. Cada array debe tener forma (dim, N_puntos).
           Si dim=2 se hace en 2D, si dim=3 se hace en 3D.
    duracion: Duración total en segundos.
    kwargs: title, xlabel, ylabel, zlabel, xlim, ylim, zlim, colors, linestyles, labels.
    """
    
    # 1. Análisis de dimensiones
    if not isinstance(datos, (list, tuple)):
        datos = [datos]
        
    dim = datos[0].shape[0]
    n_puntos = datos[0].shape[1]
    
    if dim not in [2, 3]:
        raise ValueError("La dimensión de los datos debe ser 2 o 3.")

    # 2. Cálculo de fotogramas
    frames_tot = int(duracion * fps)
    
    # 3. Configuración de la figura
    fig = plt.figure(figsize=kwargs.get('figsize', (8, 8)), dpi=kwargs.get('dpi', 120))
    if dim == 3:
        ax = fig.add_subplot(projection='3d')
        ax.set_zlabel(kwargs.get('zlabel', r'$z$'), labelpad=15)
        if 'zlim' in kwargs: ax.set_zlim(kwargs['zlim'])
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    else:
        ax = fig.add_subplot()
        
    ax.set_title(kwargs.get('title', r'Animación'))
    ax.set_xlabel(kwargs.get('xlabel', r'$x$'), labelpad=15)
    ax.set_ylabel(kwargs.get('ylabel', r'$y$'), labelpad=15)
    
    if 'xlim' in kwargs: ax.set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs: ax.set_ylim(kwargs['ylim'])
    
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # 4. Inicialización de elementos gráficos
    lineas = []
    puntos = []
    sombras = []
    
    colores = kwargs.get('colors', ['navy', 'crimson', 'darkgreen', 'darkorange'])
    linestyles = kwargs.get('linestyles', ['-'] * len(datos))
    labels = kwargs.get('labels', [f'Cuerpo {i}' for i in range(len(datos))])

    for i in range(len(datos)):
        c = colores[i % len(colores)]
        ls = linestyles[i % len(linestyles)]
        
        if dim == 3:
            linea, = ax.plot([], [], [], lw=2, color=c, linestyle=ls, label=labels[i])
            punto, = ax.plot([], [], [], 'o', color=c, markersize=5)
            sombra, = ax.plot([], [], [], '--', color='gray', alpha=0.3)
            sombras.append(sombra)
        else:
            linea, = ax.plot([], [], lw=2, color=c, linestyle=ls, label=labels[i])
            punto, = ax.plot([], [], 'o', color=c, markersize=5)
            
        lineas.append(linea)
        puntos.append(punto)

    if kwargs.get('legend', True):
        ax.legend()

    # 5. Función de actualización (Closure)
    def update(frame):
        # Mapeo del fotograma actual al índice real del array de datos
        idx = int(frame * (n_puntos - 1) / (frames_tot - 1)) if frames_tot > 1 else 0
        
        elementos_actualizados = []
        
        for i, data in enumerate(datos):
            x, y = data[0, :idx+1], data[1, :idx+1]
            
            if dim == 3:
                z = data[2, :idx+1]
                lineas[i].set_data(x, y)
                lineas[i].set_3d_properties(z)
                
                puntos[i].set_data([x[-1]], [y[-1]])
                puntos[i].set_3d_properties([z[-1]])
                
                # Proyección en la base (Z mínimo dinámico o fijo)
                z_min = kwargs.get('zlim', [np.min(data[2])])[0]
                sombras[i].set_data(x, y)
                sombras[i].set_3d_properties(np.full_like(x, z_min))
                
                elementos_actualizados.extend([lineas[i], puntos[i], sombras[i]])
            else:
                lineas[i].set_data(x, y)
                puntos[i].set_data([x[-1]], [y[-1]])
                elementos_actualizados.extend([lineas[i], puntos[i]])
                
        return elementos_actualizados

    # 6. Ejecución
    ani = FuncAnimation(
        fig, update, frames=frames_tot,
        interval=1000/fps, blit=False, repeat=False
    )

    if guardar:
        ani.save(archivo, writer="ffmpeg", fps=fps, dpi=kwargs.get('dpi', 120))
    
    plt.tight_layout()
    plt.show()
    return ani
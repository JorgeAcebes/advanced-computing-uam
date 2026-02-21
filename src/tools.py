import numpy as np

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
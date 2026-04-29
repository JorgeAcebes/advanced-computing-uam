# %%
import numpy as np
import matplotlib.pyplot as plt

# %%

D = 0.1 # Difusividad térmica de la corteza terrestre (m^2/día)
tau = 365.0 # Periodo orbital (días)
A = 10.0 # Temperatura media en superficie (Celsius)
B = 12.0 # Amplitud térmica (Celsius)
L = 20.0 # Profundidad del dominio (m)
T_fondo = 11.0 # Temperatura a 20 m bajo tierra (Celsius)
T_ini = 10.0 # Condición inicial interior (Celsius)

# Parámetros de la malla (Discretización)
N = 100 # Nodos espaciales
dx = L / N # Paso espacial
dt = 0.1 # Paso temporal (días)

# Verificación de la condición de estabilidad
alpha = D * dt / dx**2
if alpha > 0.5:
    raise ValueError("Inestabilidad numérica detectada. Reduzca dt o aumente dx.")

# Inicialización
x = np.linspace(0, L, N+1)
T = np.full(N+1, T_ini) # Manera más elegante de hacer np.ones(N+1) * T_ini
T[-1] = T_fondo    # Seteamos la temperatura a 20 metros bajo tierra a 11 grados Celsius

# Omitimos los primeros 9 años, y creamos un diccionario donde separamos 
# las estaciones (dividiendo los días por año entre 4, porque tenemos 4 estaciones)
dias_ano = int(tau)
t_inicio_captura = 9 * dias_ano
tiempos_captura = {
    t_inicio_captura + 0 * dias_ano/4: 'Primavera',
    t_inicio_captura + 1 * dias_ano/4: 'Verano',
    t_inicio_captura + 2 * dias_ano/4: 'Otono',
    t_inicio_captura + 3 * dias_ano/4: 'Invierno'
}
tiempos_keys = list(tiempos_captura.keys())
perfiles_guardados = {}

# Bucle de integración temporal (FTCS)
t = 0.0
t_fin = 10 * dias_ano
idx_captura = 0

while t <= t_fin:
    # Registramos el perfil en las estaciones
    if idx_captura < len(tiempos_keys) and t >= tiempos_keys[idx_captura]:
        etiqueta = tiempos_captura[tiempos_keys[idx_captura]]
        perfiles_guardados[etiqueta] = T.copy()
        idx_captura += 1

    # Condiciones iniciales
    T[0] = A + B * np.sin(2 * np.pi * t / tau)

    # Actualización de nodos internos (vectorizados)
    T[1:-1] += alpha * (T[2:] - 2*T[1:-1] + T[:-2])

    t += dt

# Gráficas
plt.figure(figsize=(8, 6))

colores = {'Primavera': '#1f77b4', 'Verano': '#ff7f0e', 'Otono': '#2ca02c', 'Invierno': '#d62728'}

for estacion, perfil in perfiles_guardados.items():
    plt.plot(x, perfil, label=estacion, color=colores[estacion])

plt.title('Temperatura de la corteza terrestre')
plt.xlabel('Profundidad (metros)')
plt.ylabel('T (Celsius)')
plt.legend()
plt.tight_layout()
plt.show()


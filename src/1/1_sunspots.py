import numpy as np
from pathlib import Path
ruta = Path(__file__).parent.parent / ".." / "data" / "1"

import matplotlib.pyplot as plt

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

def running_average(y_data, r=5):
    l = len(y_data)
    Y = np.zeros(l)
    for k in range(l):
        holder = []
        for m in range(-r,r+1):
            if k+m >= l or k+m < 0: continue
            holder.append(y_data[k+m])
        Y[k] = np.mean(holder)
    return Y

 
datos_sun = np.loadtxt(ruta/'sunspots.txt', float)
def sunspots_plot(datos_sun, ub = None, histograma = False, r=5):
    month = datos_sun[:,0] if ub is None else datos_sun[:ub+r,0]
    n_sunspots = datos_sun[:,1] if ub is None else datos_sun[:ub+r,1]
    year = 1749 + (month / 12)


    n_sunspots_run = running_average(n_sunspots, r=r)
    
    if ub:
        n_sunspots = n_sunspots[:ub]
        n_sunspots_run = n_sunspots_run[:ub]
        month = month[:ub]
        year = year[:ub]

    N_meses = 12
    ciclo = month% N_meses
    count = []
    for mon in range(N_meses):
        mask = ciclo == mon
        tot_month = np.sum(n_sunspots[mask])
        count.append(tot_month)


    month_names = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    _, ax = plt.subplots(figsize= (12,6), tight_layout=True) 

    ax.plot(year, n_sunspots, 'k-', label='Datos Mensuales', alpha =0.4)
    ax.plot(year, n_sunspots_run, 'r-', label=f"Running Average r={r:.0f}")
    ax.set_xlabel("Tiempo [Años]")
    ax.set_ylabel("Número de manchas solares")
    titulo = 'Detección de Manchas Solares en función del tiempo'
    ax.set_title(titulo if ub is None else f"{titulo} para los primeros {ub:.0f} puntos")
    ax.legend()

    if histograma:
        plt.figure(figsize=(10,6))
        plt.bar(month_names, count, color = 'pink')
        plt.xlabel("Mes")
        plt.ylabel("Número de manchas solares")
        plt.title("Número de manchas solares totales por mes")


    plt.show()
    plt.close()
    return year, n_sunspots, n_sunspots_run, count

sunspots_plot(datos_sun, histograma = True);
sunspots_plot(datos_sun, ub = 10**3);

 



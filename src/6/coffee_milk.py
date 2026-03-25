# %% Imports y Declaraciones
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import linregress
import sys
from pathlib import Path
import importlib
from matplotlib.ticker import MaxNLocator

figuras = Path(__file__).resolve().parent.parent / '6' / 'figures'
figuras.mkdir(parents=True, exist_ok=True)  

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
import tools as ja
importlib.reload(ja)
ja.setup_style(base_size=19, dpi=120)

np.random.seed(19)

# === Parámetros libres del código ===
determinista = 1
random_walk  = 1
testing = 0
save_figs = 0


# == Variables y Funciones ==

Long = np.array([50, 50])
Nx, Ny = Long * 2 + 1  # Número puntos grid en x, y

Nt = 2500
T_max = 100.0
dt = T_max / Nt

Lx, Ly = Long  # Longitud del dominio
dx, dy = 2 *Lx / (Nx - 1), 2 *Ly / (Ny - 1)
D = (dx**2) / (4 * dt) # Coeficiente de difusión para que coincida con RW
L0 = 5 # Semi anchura de la "delta"
N_particulas = 5 * 10**5 if not testing else 10**2


bins = 20 # Para el histograma en random walks 
times = np.array([0, 0.01, 0.05, 1]) # Tiempos en los que graficaremos el estado del sistema

colores = ['black', 'purple', 'teal', 'yellow'] 

def diff_fin_2D(u, rx, ry):
    """Actualización determinista para difusión 2D."""
    u[1:-1, 1:-1] += (
        rx * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) +
        ry * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])
    )

    # Condiciones de contorno de Neumann homogéneas
    u[0, :] = u[1, :]       
    u[-1, :] = u[-2, :]     
    u[:, 0] = u[:, 1]       
    u[:, -1] = u[:, -2]     

    return u


def diffusion(L, Nt, T_max, N_particulas=10000):

    valid_nodes = np.arange(-nx_0, nx_0+1) * dx
    x0, y0 = np.random.choice(valid_nodes, size=N_particulas), np.random.choice(valid_nodes, size=N_particulas)

    pos = np.column_stack((x0,y0))

    S_t = np.zeros(Nt+1)
    saved_pos = {}
    histogram = {}
    instantes_c = times * Nt
    S_max = np.log(bins * bins)

    H, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=bins, range=[[-L, L], [-L, L]], density=True)
    saved_pos[0] = pos.copy()
    histogram[0] = H.copy()

    for n in np.arange(1, Nt+1):

        mov = np.random.choice([-1, 1], size=N_particulas)
        eje = np.random.randint(0,2, size=N_particulas)

        pos[np.arange(N_particulas), eje] += mov
        # Condiciones de contorno reflectantes
        pos = np.where(pos > L, 2*L - pos, pos)
        pos = np.where(pos < -L, -2*L - pos, pos)

        # Cálculo de probabilidades y Entropía de Shannon
        H_brut, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=bins, range=[[-L, L], [-L, L]], density = False)
        P = (H_brut / N_particulas).flatten()
        P = P[P > 0]  
        S_t[n] = -np.sum(P * np.log(P))

        # Guardar posiciones
        if n in instantes_c :
            saved_pos[n] = pos.copy()
            H, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=bins, range=[[-L, L], [-L, L]], density=True)
            histogram[n] = H.copy()

    # Determinación empírica del tiempo de equilibrio (95% de saturación)
    t = np.linspace(0, T_max, Nt+1)
    t_eq = np.argmax(S_t > 0.95 * S_max) 
    if t_eq == 0: t_eq = Nt # Si la difusión no saturó
    t_eq = t[t_eq]

    return S_t, t_eq, saved_pos, histogram


# --- Modelos determinista y random walk ---


if determinista:
    # --- Parámetros físicos y numéricos ---
    b = D * dt * (1/dx**2 + 1/dy**2)
    rx = D * dt / dx**2
    ry = D * dt / dy**2

    x = np.linspace(-Lx, Lx, Nx)
    y = np.linspace(-Ly, Ly, Ny)

    X, Y = np.meshgrid(x, y, indexing='ij') 

    # Estabilidad de Von Neumann para 2D
    if b > 0.5:
        raise ValueError(f"Inestabilidad: beta = {b:.3f} > 0.5. Aumenta: Lx, Ly, Nt, o Disminuye: D, T_max, Nx, Ny, ")

    u = np.zeros((Nx, Ny))

    # Condición inicial
    cx, cy = int(Nx/2), int(Ny/2)
    nx_0 = max(1, int(L0 / dx)) 
    ny_0 = max(1, int(L0 / dy))

    u[cx-nx_0:cx+nx_0+1, cy-ny_0:cy+ny_0+1] = 1.0
    
    u /= (np.sum(u) * dx * dy) # Normalizamos para que la densidad refleje la probabilidad
    
    print(f'El centro de la delta está en {cx,cy} y tiene un ancho {2*L0} x {2*L0}')
    # --- Gráficas ---

    fig, ax  = plt.subplots(figsize=(8, 5))
    figcmap, axcmap  = plt.subplots(2,2, figsize=(10, 10))
    axcmap = axcmap.flatten()
    fig_3d = plt.figure(figsize=(10,10))
    ax_3d = fig_3d.add_subplot(projection='3d')


    # Ploteamos estado inicial
    i = 0
    ax.plot(x, u[:, cy], label="t = 0.00", color=colores[i], linestyle="-.")
    im = axcmap[i].imshow(u.T, origin='lower', extent=[-Lx, Lx, -Ly, Ly], aspect='auto', cmap='viridis')
    axcmap[i].text(0.95, 0.95, f"$t = {0.00:.2f}\\,$s", color="white", ha="right", va="top", transform=axcmap[i].transAxes, fontsize=16)
    alpha = 0.1
    ax_3d.plot_surface(X, Y, u, alpha = alpha, label =f"$t = {0.00:.2f}\\,$s", color=colores[i])


    snapshots = times * Nt

    # Evolución temporal
    for n in np.arange(1, Nt+1):
        u = diff_fin_2D(u, rx, ry)    
        
        if n in snapshots:
            i += 1
            ax.plot(x, u[:, cy], label=f"t = {n*dt:.2f}", color = colores[i])
            axcmap[i].imshow(u.T, origin='lower', extent=[-Lx, Lx, -Ly, Ly], aspect='auto', cmap='viridis')
            axcmap[i].text(0.95, 0.95, f"$t = {n*dt:.2f}\\,$s", color="white", ha="right", va="top", transform=axcmap[i].transAxes, fontsize=16)
            ax_3d.plot_surface(X, Y, u, alpha = alpha, label =f"$t = {n*dt:.2f}\\,$s", color = colores[i])
            alpha += 0.05

    ax.set_xlabel("Posición $x$")
    ax.set_ylabel("Densidad $u(x, y=0.5,t)$")
    ax.set_title("Perfil transversal de la difusión 2D")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title = r'Tiempo $[s]$',title_fontsize=14)
    plt.grid(True)

    axcmap[0].set_ylabel("$y$")
    axcmap[2].set_ylabel("$y$")
    axcmap[2].set_xlabel("$x$")
    axcmap[3].set_xlabel("$x$")
    cbar = figcmap.colorbar(im, ax=axcmap.ravel().tolist(), fraction=0.04, pad=0.04)
    cbar.set_label(r"Densidad $u(x,y,t)$")
    figcmap.suptitle(r"Dinámica de la difusión 2D", fontsize=30)

    ax_3d.set_xlabel(r"$x$", labelpad=15)
    ax_3d.set_ylabel(r"$y$", labelpad=15)
    ax_3d.set_zlabel(r"Densidad $u(x,y,t)$", labelpad = 25)
    ax_3d.legend(loc="upper right")
    ax_3d.set_title(r"Dinámica de la difusión 2D", fontsize=30)

    ax_3d.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_3d.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_3d.zaxis.set_major_locator(MaxNLocator(nbins=5))

    if save_figs:
        fig.savefig("figures/diffusion_perfil_determ.png", dpi=300, bbox_inches='tight')
        figcmap.savefig("figures/diffusion_cmap_determ.png", dpi=300, bbox_inches='tight')
        fig_3d.savefig("figures/diffusion_3d_determ.png", dpi=300)

    plt.grid(False)
    plt.show()


if random_walk:
    # --  Variables y Funciones --
    taza_size = [20, 35, int(np.mean(Long))]
    resultados = {} # Inicializamos diccionario donde recoger resultados de diffusion
    t_plot = np.linspace(0, T_max, Nt+1)

    # -- Gráficos --
    fig, axs = plt.subplots(1, 3, figsize=(22, 5.5))

    # Evolución de la entropía
    for L in taza_size:
        resultados[L] = diffusion(L, Nt, T_max, N_particulas=N_particulas)
        axs[0].plot(t_plot, resultados[L][0], label=rf"$L = {L}$")
    axs[0].axhline(np.log(bins**2), color='k', linestyle='--', label=rf"$S_{{\max}} = \ln({bins}^2)$")   
    axs[0].set_xlabel(r"Tiempo $t")
    axs[0].set_ylabel(r"Entropía $S(t)$")
    axs[0].legend()
    axs[0].set_title(r"Evolución entrópica")

    # Tiempo de equilibrio vs Cuadrado del tamaño taza
    L_vals = np.array(taza_size)
    L2_vals = L_vals**2
    t_eq_vals = np.array([resultados[L][1] for L in taza_size])

    res = linregress(L2_vals, t_eq_vals)
    axs[1].plot(L2_vals, t_eq_vals, 'ko', markersize=8)
    axs[1].plot(L2_vals, res.intercept + res.slope * L2_vals, 'r--', 
                label=rf"$t_{{eq}} ={res.slope:.4f} L^2 {res.intercept:.4f} $)")
    axs[1].set_xlabel(r"Tamaño al cuadrado $L^2$")
    axs[1].set_ylabel(r"Tiempo de equilibrio $t_{eq}$")
    axs[1].legend()
    axs[1].set_title(r"Escalamiento termodinámico")

    # Posición de partículas para taza fija
    L_c = taza_size[2]
    pos_c = resultados[L_c][2]
    hist_c = resultados[L_c][3]

    figcmap, axcmap  = plt.subplots(2,2, figsize=(10, 10))
    figcmap.suptitle(r"Dinámica de la difusión 2D", fontsize=30)
    axcmap = axcmap.flatten()
    
    alpha = 0.8
    zorder = 10
    for i, t in enumerate(pos_c.keys()): 
        axs[2].scatter(pos_c[t][:, 0], pos_c[t][:, 1], s=0.5, alpha=alpha, color=colores[i], label=rf"$t = {t_plot[t]}$", zorder = zorder)
        axcmap[i].imshow(hist_c[t].T, origin='lower', extent=[-L_c, L_c, -L_c, L_c], aspect='auto', cmap='viridis')
        axcmap[i].text(0.95, 0.95, f"$t = {t_plot[t]:.2f}\\,$s", color="white", ha="right", va="top", transform=axcmap[i].transAxes, size=16)
        alpha -= 0.25
        zorder -= 2
        if i == 0:
            im = axcmap[i].imshow(hist_c[t].T, origin='lower', extent=[-L_c, L_c, -L_c, L_c], aspect='auto', cmap='viridis')

    axcmap[0].set_ylabel("$y$")
    axcmap[2].set_ylabel("$y$")
    axcmap[2].set_xlabel("$x$")
    axcmap[3].set_xlabel("$x$")
    cbar = figcmap.colorbar(im, ax=axcmap.ravel().tolist(), fraction=0.04, pad=0.04)
    cbar.set_label(r"Densidad $u(x,y,t)$")


    axs[2].set_xlim(-L_c, L_c)
    axs[2].set_ylim(-L_c, L_c)
    axs[2].set_aspect('equal')
    axs[2].set_xlabel(r"$x$")
    axs[2].set_ylabel(r"$y$")
    axs[2].legend(loc='upper right', markerscale=10)
    axs[2].set_title(rf"Dispersión ($L={L_c}$)")

    if save_figs:
        fig.savefig("figures/multiplot_analysis_rw.png", dpi=300, bbox_inches='tight')
        figcmap.savefig("figures/diffusion_cmap_rw.png", dpi=300, bbox_inches='tight')

    plt.show()

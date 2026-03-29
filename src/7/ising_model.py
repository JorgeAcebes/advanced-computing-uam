# %% Imports y Declaraciones
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import linregress
import sys
from pathlib import Path
import importlib
from matplotlib.ticker import MaxNLocator

figuras = Path(__file__).resolve().parent.parent / '7' / 'figures'
figuras.mkdir(parents=True, exist_ok=True)  

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
import tools as ja
importlib.reload(ja)
ja.setup_style(base_size=19, dpi=120)

np.random.seed(19)

# === Parámetros libres del código ===
testing = 0
save_figs = 0


# Variables globales

red = 10 # tamaño de la red de espines
T = 0.25 # temperatura de la red

dH = 0.5 # incremento del paso de histéresis

J = 1/2
mu = 1













# if save_figs:
#     fig.savefig("figures/multiplot_analysis_rw.png", dpi=300, bbox_inches='tight')
#     figcmap.savefig("figures/diffusion_cmap_rw.png", dpi=300, bbox_inches='tight')

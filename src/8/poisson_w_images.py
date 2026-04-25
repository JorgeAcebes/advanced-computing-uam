# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import cv2

path_padre = Path(__file__).resolve().parent.parent
sys.path.append(str(path_padre))
from tools import solve_poisson_sor
from tools import setup_style
from matplotlib.ticker import MaxNLocator

setup_style()

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::
# decidir qué imágenes se emplean:
uam = 1
mapache = 1
hamburgo = 1
excercise_9_1 = 1
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::

# =====================================================
# Ruta hacia la carpeta de data
current_dir = Path(__file__).resolve().parent
DATA_8_DIR = current_dir.parent.parent / 'data' / '8'

# Ruta de figuras
figuras = Path(__file__).resolve().parent.parent / '8' / 'figures'
figuras.mkdir(parents=True, exist_ok=True)  
# =====================================================

def process_image(image_path, dx=0.1, tol=1e-3, L=None, mode='dirichlet', rho_scale=1.0, kernel = 5):
    """
    mode : {'dirichlet', 'charge', 'both'}
        'dirichlet': contornos como conductores a potencial fijo.
        'charge'   : áreas encerradas inyectan densidad de carga rho; los bordes exteriores siguen siendo tierra (Dirichlet = 0).
        'both'     : figuras abiertas: Dirichlet, figuras cerradas: densidad de carga.

    rho_scale: Factor multiplicativo sobre rho en los modos 'charge' y 'both'.
    """

    try:
        img_array = np.fromfile(image_path, np.uint8) # Obtenemos array 1D que representa los bytes de la imagen
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE) # Leemos la imagen en escala blanco-negro
    except Exception as e:
        raise ValueError(f"Fallo en la lectura física del archivo: {e}")
    
    if img is None:
        raise ValueError(f"OpenCV no pudo decodificar la imagen: {image_path}")
    print('OpenCV ha podido decodificar la imagen')

    blurred = cv2.GaussianBlur(img, (kernel, kernel), 1) # Aplicamos un filtro Gaussiano para suavizar bordes

    print('Detectando bordes...')
    edges = cv2.Canny(blurred, threshold1=10, threshold2=120) # Detectamos bordes
    print('¡Bordes detectados!')

    print('Extrayendo la topología...')
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('Topología extraída!')

    N_y, N_x = img.shape
    if L: # Refactorizamos el diferencial a y la extensión de la imagen para que la imagen cuadre con las dimensiones
        L = np.abs(L)
        a = L / (N_x - 1)
        extent = [0, L, 0, L]
    else:
        a = dx
        extent = None

    V = np.zeros((N_y, N_x), dtype=np.float64) 
    rho = np.zeros((N_y, N_x), dtype=np.float64)
    is_boundary = np.zeros((N_y, N_x), dtype=bool)

    # Tierra en el borde exterior (problema de Dirichlet bien planteado)
    is_boundary[0, :] = is_boundary[-1, :] = True
    is_boundary[:, 0] = is_boundary[:, -1] = True

    AREA_THRESHOLD = 50.0
    mask = np.zeros_like(img)

    print('Definiendo el potencial / densidad de carga según topología...')
    for cnt in contours:
        area = cv2.contourArea(cnt) # Obtenemos el área en píxeles que delimitan los contours
        is_closed = area > AREA_THRESHOLD

        cnt_mask = np.zeros((N_y, N_x), dtype=np.uint8)
        if is_closed:
            cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        else:
            cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=1)

        mean_intensity = cv2.mean(img, mask=cnt_mask)[0]
        is_conductor   = cnt_mask > 0

        use_dirichlet = (
            mode == 'dirichlet'
            or (mode == 'both' and not is_closed)
        )
        use_charge = (
            mode == 'charge'
            or (mode == 'both' and is_closed)
        )

        if use_dirichlet:
            # I(0): V = +1 ;  I(255): V = −1
            v_pot = 1.0 - (2.0 * mean_intensity / 255.0)
            is_boundary[is_conductor] = True
            V[is_conductor] = v_pot

        elif use_charge:
            # I(0): rho_scale ;  I(255): −rho_scale
            rho_val = rho_scale * (1.0 - 2.0 * mean_intensity / 255.0)
            rho[is_conductor] = rho_val
            # No se toca is_boundary: la región es libre, no un conductor

        mask = cv2.bitwise_or(mask, cnt_mask)

    print('Condiciones definidas.')

    print('Resolviendo la ecuación de Poisson... (proceso lento)')
    V_solved = solve_poisson_sor(V, rho, is_boundary, a, omega=1.9, tol=tol)
    print('¡Poisson resuelta!')

    return V_solved, mask, extent


def plot_potential(V, title, mask, save_fig_title = 'Título', extent=None, origin='upper', no_ticks=False):
    plt.imshow(V, cmap='coolwarm', origin='upper', extent=extent)
    plt.colorbar(label=r'Potencial Electrostático $V$ [V]')
    plt.contour(mask, levels=[127], colors='white', linewidths=0.5, alpha=0.5,origin=origin, extent=extent)
    plt.title(title)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    if no_ticks:
        plt.xticks([])
        plt.yticks([])
    plt.savefig(figuras/f'{save_fig_title}.png', bbox_inches='tight', dpi = 300) 
    plt.show()


for tol_title in [0.1, 0.01, 0.001]:
    tol = int(np.log10(tol_title))

    if uam:
        uam_dir = DATA_8_DIR / 'uam.png'
        V_uam, mask_uam, _ = process_image(str(uam_dir), mode='both', kernel=5, tol=tol_title)
        plot_potential(V_uam, '', mask_uam, save_fig_title=f'uam_{tol_title}', origin='lower', no_ticks=True)

    if mapache:
        mapache_dir = DATA_8_DIR / 'mapache.png'
        V_mapache, mask_mapache, _ = process_image(str(mapache_dir), mode='dirichlet', tol=tol_title)
        plot_potential(V_mapache, '', mask_mapache, save_fig_title=f'Mapache_{tol_title}', origin='lower', no_ticks=True)

    if hamburgo: 
        ham_dir = DATA_8_DIR / 'hamburgo.png'
        V_ham, mask_ham, _ = process_image(str(ham_dir), mode='dirichlet', tol=tol_title)
        plot_potential(V_ham, '', mask_ham, save_fig_title=f'Hamburgo_{tol_title}', origin='lower', no_ticks=True)


if excercise_9_1:
    dir = DATA_8_DIR / '9_1.png'
    V, mask, extent = process_image(str(dir), tol=1e-6, L=100, mode='charge')
    plot_potential(V/10000, '', mask, save_fig_title='poisson_9_1_image', extent=extent) # Divido entre 10.000 para que case con la densidad del potencial del ejercicio 9.1

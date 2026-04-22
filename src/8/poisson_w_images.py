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

setup_style()



# :::::::::::::::::::::::::::::::::::::::::::::::::::::::
# decidir qué imágenes se emplean:
mapache = 0
hamburgo = 0
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



def process_image(image_path, dx=0.1, tol = 1e-3, L = None):
    """
    Lee una imagen PNG, detecta contornos topológicos y resuelve 
    la ecuación de Laplace en el dominio bidimensional resultante.

    L: longitud en x, y que queremos que tenga la imagen (va a ser siempre ratio 1:1)
    """
    # 1. Carga y preprocesamiento de la imagen
    try:
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        raise ValueError(f"Fallo en la lectura física del archivo: {e}")

    if img is None:
        raise ValueError(f"OpenCV no pudo decodificar la imagen: {image_path}")
    else:
        print('OpenCV ha podido decodificar la imagen')

    # Suavizado gaussiano para mitigar el ruido de alta frecuencia
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 2. Detección de bordes mediante el algoritmo de Canny
    print('Detectando bordes...')
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    print('¡Bordes detectados!')

    # 3. Extracción de la topología (Contornos)
    # RETR_EXTERNAL ignora agujeros internos, quedándose con la envolvente principal
    print('Extrayendo la topología...')
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('Topología extraída!')

    N_y, N_x = img.shape
    if L:
        L = np.abs(L)
        a = L / (N_x - 1)
        extent = [0, L, 0, L]
    else:
        a = dx
        extent = None
    V = np.zeros((N_y, N_x), dtype=np.float64)
    is_boundary = np.zeros((N_y, N_x), dtype=bool)
    
    # Condición de contorno exterior (Tierra: 0V) para asegurar que el problema de Dirichlet 
    # esté bien planteado y el método iterativo converja.
    is_boundary[0, :] = is_boundary[-1, :] = is_boundary[:, 0] = is_boundary[:, -1] = True
    
    
    # 4. Clasificación topológica y Mapeo de potencial por escala de grises
    AREA_THRESHOLD = 50.0  
    
    # Máscara global para la visualización final
    mask = np.zeros_like(img)
    
    print('Definiendo el potencial según la topología y radiometría...')
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Máscara local aislada para el contorno actual
        cnt_mask = np.zeros((N_y, N_x), dtype=np.uint8)
        
        if area > AREA_THRESHOLD:
            # Figura topológicamente cerrada
            cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        else:
            # Figura abierta (conductor unidimensional)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=1)
            
        # Extracción de la intensidad media de la figura en la imagen original
        mean_intensity = cv2.mean(img, mask=cnt_mask)[0]
        
        # Transformación lineal: I(0) -> V(+1) ; I(255) -> V(-1)
        v_pot = 1.0 - (2.0 * mean_intensity / 255.0)
        
        # 5. Mapeo del dominio a las matrices del solucionador
        is_conductor = cnt_mask > 0
        is_boundary[is_conductor] = True
        V[is_conductor] = v_pot  # Condición de Dirichlet específica para la figura
        
        # Integración en la máscara global para el ploteo
        mask = cv2.bitwise_or(mask, cnt_mask)
            
    print('Potencial definido rigurosamente.')
    
    # Densidad de carga nula para el vacío restante
    rho = np.zeros_like(V)

    print('Resolviendo la ecuación de Poisson... (proceso lento)')
    V_solved = solve_poisson_sor(V, rho, is_boundary, a, omega=1.9, tol=tol)
    print('Poisson resuelta!')

    return V_solved, mask, extent

def plot_potential(V, title, mask, save_fig = False, extent=None):
    plt.imshow(V, cmap='coolwarm', origin='upper', extent=extent)
    plt.colorbar(label='Potencial (V)')
    plt.contour(mask, levels=[127], colors='white', linewidths=0.5, alpha=0.5,origin='upper', extent=extent)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(figuras/f'{title}.png', bbox_inches='tight', dpi = 300) 
    plt.show()


if mapache:
    mapache_dir = DATA_8_DIR / 'mapache.png'
    V_mapache, mask_mapache = process_image(str(mapache_dir))
    plot_potential(V_mapache, 'Mapache', mask_mapache)

if hamburgo: 
    ham_dir = DATA_8_DIR / 'hamburgo.png'
    V_ham, mask_ham = process_image(str(ham_dir))
    plot_potential(V_ham, 'Dos personas en Hamburgo', mask_ham)


if excercise_9_1:
    dir = DATA_8_DIR / '9_1.png'
    V, mask, extent = process_image(str(dir), tol=1e-6, L=100)
    plot_potential(V, 'Resolución ejercicio 9.1', mask, save_fig=1, extent=extent)

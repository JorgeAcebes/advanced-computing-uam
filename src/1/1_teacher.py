import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace, sin, exp, pi, loadtxt
from pathlib import Path
ruta = Path(__file__).parent.parent / ".." / "data" / "1"
import vpython as vp


#  Oscilador amortiguado

x1, x2 = linspace(0.0, 2.0, 20), linspace(0.0, 2.0, 200)
y1, y2, y3 = exp(-x1), exp(-x2), sin(2*pi*x2)
y4 = y2*y3

l1, l3, l4 = plt.plot(x1, y1, "bD-"), plt.plot(x2, y3, "go-"), plt.plot(x2, y4, "rs-")
plt.ylim(-1.1, 1.1)
plt.xlabel("Segundos")
plt.ylabel("Voltios")
plt.legend(  (l3[0], l4[0]), ("Oscilatorio", "Amortiguado"), shadow = True)
plt.title("Movimiento Oscilatorio Amortiguado")
plt.show()
plt.close()

 

#  Diagrama HR
datos = loadtxt(ruta/"stars.txt", float)
    
x,y = datos[:,0], datos[:,1]

plt.figure(figsize=(10, 6))
plt.scatter(x,y, c=x, cmap= 'hsv', s=75, edgecolors='k')
plt.xlabel("Temperatura")
plt.ylabel("Magnitud")
plt.xlim(13000, 0)
plt.ylim(20, -5)
plt.text(4500, 4.5, "Secuencia Principal", fontsize=14, style="italic")
plt.text(11000, 15, "Enanas Blancas", fontsize = 14, style = "italic")
plt.title("Diagrama Hertzsprung-Russell")
plt.show()

#  Jet

datos = loadtxt(ruta/"circular.txt", float)
plt.imshow(datos), plt.jet(), plt.colorbar()
plt.show()

 

# Rotating Ball

vp.canvas(x = 500, y=200, width = 500, height = 500, center = vp.vector(0,0,1), forward = vp.vector(0,0,-1), background = vp.color.blue, foreground = vp.vector(1,1,0))
s = []
s.append(vp.sphere(pos = vp.vector(1,0,0), radius = 0.25, color = vp.color.yellow))
for theta in np.arange(0,10*vp.pi, 0.1):
    vp.rate(30)
    x = np.cos(theta)
    y = np.sin(theta)
    s[0].pos = vp.vector(x,y,0)
while True:
    vp.rate(30)

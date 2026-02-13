import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
ruta = Path(__file__).parent.parent / ".." / "data" / "1"
import vpython as vp


L,M,N = 5,4,3
RNa, RCl = 0.5, 0.4
colNa, colCl = vp.color.green, vp.color.magenta
for k in range(-N,N+1,2):
    for j in range(-M,M+1,2):
        for i in range(-L,L+1,2):
            vp.sphere(pos=vp.vector(i,j,k), radius = RNa, color = colNa)
        for i in range(-L,L+1,2):
            vp.sphere(pos=vp.vector(i,j,k), radius = RCl, color = colCl)
    for j in range(-M+1, M+1, 2):
        for i in range(-L, L+1, 2):
            vp.sphere(pos=vp.vector(i,j,k), radius = RCl, color = colCl)
        for i in range(-L+1, L+1, 2):
            vp.sphere(pos=vp.vector(i,j,k), radius = RNa, color = colNa)
for k in range(-N+1, N+1, 2):
    for j in range(-M,M+1,2):
        for i in range(-L,L+1,2):
            vp.sphere(pos=vp.vector(i,j,k), radius = RCl, color = colCl)
        for i in range(-L,L+1,2):
            vp.sphere(pos=vp.vector(i,j,k), radius = RNa, color = colNa)
    for j in range(-M+1, M+1, 2):
        for i in range(-L, L+1, 2):
            vp.sphere(pos=vp.vector(i,j,k), radius = RNa, color = colNa)
        for i in range(-L+1, L+1, 2):
            vp.sphere(pos=vp.vector(i,j,k), radius = RCl, color = colCl)
while True:
    vp.rate(30)
 
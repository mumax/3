from mumax2 import *
from math import *

# Standard Problem 4

Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)

load('micromagnetism')

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1)
setv('dt', 1e-15)
setv('m_maxerror', 1e-4)
x = 1/sqrt(3)
m=[ [[[x]]], [[[x]]], [[[x]]] ]
setarray('m', m)

save('m', 'dump', [])
save('H_eff', 'dump',[])
save('H_ex', 'dump',[])
save('torque', 'dump',[])

autosave("m", "dump", [], 100e-12)
autotabulate(["t", "<m>", "m_error", "m_peakerror", "badsteps", "dt", "maxtorque"], "m.txt", 10e-12)

run(2e-9)

setv('alpha', 0.02)
setv('dt', 1e-15)
Bx = -24.6E-3
By =   4.3E-3
Bz =   0      
setv('B_ext', [Bx, By, Bz])

run(1e-9)

sync()

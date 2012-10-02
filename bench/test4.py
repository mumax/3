from mumax2 import *

# Standard Problem 4

Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)

load('llg')
load('demag')
load('exchange6')
load('solver/euler')

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1)
setv('dt', 1e-15)

m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)


savegraph("graph.png")

#run(2e-9) #relax
setv('alpha', 0.02)
setv('dt', 1e-15)
setv('t', 0)

steps(5000)

printstats()

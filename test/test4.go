package main

import(
	. "nimble-cube/core"
	"nimble-cube/mag"
	"nimble-cube/gpu/conv"
)

func main(){
	N0, N1, N2 := 1, 32, 128
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9	
	mesh := NewMesh(N0, N1, N2, cx, cy, cz) 

	acc := 8
	kernel := mag.BruteKernel(mesh.ZeroPadded(), acc)
	demag := conv.NewBasic(mesh.GridSize(), kernel)

	Log(demag)
}

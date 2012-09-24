package main

import (
	. "nimble-cube/core"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
)

func main() {
	N0, N1, N2 := 1, 32, 128
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := NewMesh(N0, N1, N2, cx, cy, cz)
	size := mesh.GridSize()

	m1 := MakeChan3(size)
	hd := MakeChan3(size)

	acc := 8
	kernel := mag.BruteKernel(mesh.ZeroPadded(), acc)
	go conv.NewSymmetricHtoD(size, kernel, m1.MakeRChan3(), hd).Run()

	Msat := 1.0053
	aex := Mu0 * 13e-12 / Msat
	hex := MakeChan3(size)
	go mag.NewExchange6(m1.MakeRChan3(), hex, mesh, aex).Run()

	heff := MakeChan3(size)
	go NewAdder3(heff, hd.MakeRChan3(), hex.MakeRChan3()).Run()

	const alpha = 0.02
	torque := MakeChan3(size)
	go mag.RunLLGTorque(torque, m1.MakeRChan3(), heff.MakeRChan3(), alpha)

//	m2 := MakeChan3(size)
//	solver := NewEuler(m2, m1, )

}

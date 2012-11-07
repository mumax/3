package main

import (
	"fmt"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
	. "nimble-cube/nimble"
	"os"
	"time"
)

func main() {
	N0, N1, N2 := IntArg(0), IntArg(1), IntArg(2)
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := NewMesh(N0, N1, N2, cx, cy, cz)
	size := mesh.GridSize()
	Log("mesh:", mesh)
	Log("block:", BlockSize(mesh.GridSize()))

	m := MakeChan3(size, "m")
	hd := MakeChan3(size, "Hd")

	acc := 1
	kernel := mag.BruteKernel(mesh.ZeroPadded(), acc)
	Stack(conv.NewSymmetricHtoD(size, kernel, m.MakeRChan3(), hd))

	Msat := 1.0053
	aex := Mu0 * 13e-12 / Msat
	hex := MakeChan3(size, "Hex")
	Stack(mag.NewExchange2D(m.MakeRChan3(), hex, mesh, aex))

	heff := MakeChan3(size, "Heff")
	Stack(NewAdder3(heff, hd.MakeRChan3(), hex.MakeRChan3()))

	const alpha = 1
	torque := MakeChan3(size, "τ")
	Stack(mag.NewLLGTorque(torque, m.MakeRChan3(), heff.MakeRChan3(), alpha))

	const dt = 100e-15
	solver := mag.NewEuler(m, torque.MakeRChan3(), dt)
	mag.SetAll(m.UnsafeArray(), mag.Uniform(0, 0.1, 1))

	RunStack()

	// run for about 10s and output time/step
	start := time.Now()
	duration := time.Since(start)
	const N = 20
	var steps int64
	solver.Steps(1)
	for duration < 2*time.Second {
		solver.Steps(N)
		steps += N
		duration = time.Since(start)
	}

	fmt.Println(N0, N1, N2, *Flag_maxblocklen, duration.Nanoseconds()/(1e3*steps))

	ProfDump(os.Stdout)
	Cleanup()
}

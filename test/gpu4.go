package main

import (
	"code.google.com/p/nimble-cube/gpu"
	"code.google.com/p/nimble-cube/mag"
	"code.google.com/p/nimble-cube/nimble"
	"code.google.com/p/nimble-cube/uni"
	"fmt"
	"math"
	//"os"
)

// Standard problem 4 on GPU
func main() {
	nimble.Init()
	defer nimble.Cleanup()
	nimble.SetOD("gpu4.out")

	mem := nimble.GPUMemory

	const (
		N0, N1, N2 = 1, 32, 128
		cx, cy, cz = 3e-9, 3.125e-9, 3.125e-9
		Bsat       = 1.0053
		Aex_red    =  mag.Mu0 * 13e-12 / Bsat
		α          = 1
	)

	mesh := nimble.NewMesh(N0, N1, N2, cx, cy, cz)
	fmt.Println("mesh:", mesh)

	// TODO: MakeChanN -> NewQuant()
	m := nimble.MakeChanN(3, "m", "", mesh, mem, 0)
	M := gpu.Device3(m.ChanN().UnsafeData())
	M[0].Memset(0)
	M[1].Memset(1 / math.Sqrt2)
	M[2].Memset(1 / math.Sqrt2)

	acc := 8
	kernel := mag.BruteKernel(mesh, acc)
	B := gpu.NewConvolution("B", "T", mesh, mem, kernel, m).Output()

	exch := gpu.NewExchange6("Bex", m, Aex_red)
	nimble.Stack(exch)
	Bex := exch.Output()

	BeffBox := gpu.NewSum("Beff", B, Bex, Bsat, 1, mem)
	Beff := BeffBox.Output()

	tBox := gpu.NewLLGTorque("torque", m, Beff, α)
	nimble.Stack(tBox)
	torque := tBox.Output()

	solver := gpu.NewHeun(m, torque, 1e-15, mag.Gamma0)
	solver.Maxerr = 1e-4
	solver.Maxdt = 1e-12

	every := 100
	uni.Autosave(m, every, gpu.GPUDevice)
	uni.Autotable(m, every/10, gpu.GPUDevice)

	solver.Advance(1e-9)

	var avg [3]float32
	for i := range avg {
		avg[i] = gpu.Sum(m.UnsafeData()[i].Device(), 0) / float32(mesh.NCell())
	}
	fmt.Println("avg:", avg)

	const (
		Bx = -24.6E-3
		By = 4.3E-3
		Bz = 0
	)
	Bext := gpu.RunConst("Bext", "T", mesh, mem, []float64{Bz, By, Bx})
	BeffBox.MAdd(Bext, 1)
	tBox.Alpha = 0.02
	solver.Advance(1e-9)

	for i := range avg {
		avg[i] = gpu.Sum(m.UnsafeData()[i].Device(), 0) / float32(mesh.NCell())
	}
	fmt.Println("avg:", avg)
}

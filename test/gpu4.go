package main

import (
	"code.google.com/p/mx3/gpu"
	"code.google.com/p/mx3/mag"
	"code.google.com/p/mx3/nimble"
	"fmt"
	"math"
	"os"
)

// Standard problem 4 on GPU
func main() {
	nimble.Init()
	defer nimble.Cleanup()
	nimble.SetOD("gpu4.out")

	mem := nimble.GPUMemory

	const (
		N0, N1, N2 = 1, 32, 128
		Sx, Sy, Sz = 3e-9, 125e-9, 500e-9
		cx, cy, cz = Sx / N0, Sy / N1, Sz / N2
		Bsat       = 800e3 * mag.Mu0
		Aex_red    = 13e-12 / (Bsat / mag.Mu0)
		α          = 1
	)

	mesh := nimble.NewMesh(N0, N1, N2, cx, cy, cz)
	fmt.Println("mesh:", mesh)

	// TODO: MakeChanN -> NewQuant()
	m := nimble.MakeChanN(3, "m", "", mesh, mem, 0)
	M := gpu.Device3(m.ChanN().UnsafeData())
	M[0].Memset(float32(1 / math.Sqrt(3)))
	M[1].Memset(float32(1 / math.Sqrt(3)))
	M[2].Memset(float32(1 / math.Sqrt(3)))

	acc := 5.
	kernel := mag.BruteKernel(mesh, acc)
	B := gpu.NewConvolution("B", "T", mesh, mem, kernel, m).Output()
	Bex := gpu.NewExchange6("Bex", m, Aex_red).Output()

	BeffBox := gpu.NewSum("Beff", B, Bex, Bsat, 1, mem)
	Beff := BeffBox.Output()

	tBox := gpu.NewLLGTorque("torque", m, Beff, α)
	torque := tBox.Output()

	solver := gpu.NewHeun(m, torque, 10e-15, mag.Gamma0)
	solver.Maxerr = 2e-4
	solver.Mindt = 1e-15

	//every := 100
	//uni.Autosave(m, every, gpu.GPUDevice)
	//uni.Autotable(m, every/10, gpu.GPUDevice)

	solver.Advance(2.5e-9)
	//solver.Relax(1e-7)

	var avg [3]float32
	for i := range avg {
		avg[i] = gpu.Sum(m.UnsafeData()[i].Device(), 0) / float32(mesh.NCell())
	}
	want := [3]float32{0, 0.12572803, 0.96669436}

	err := math.Sqrt(float64(sqr(avg[0]-want[0]) + sqr(avg[1]-want[1]) + sqr(avg[2]-want[2])))
	fmt.Println("avg:", avg, "err:", err)
	if err > 1e-3 {
		fmt.Println("FAILED")
		//os.Exit(2)
	}
	fmt.Println("OK")

	const (
		Bx = -24.6E-3
		By = 4.3E-3
		Bz = 0
	)
	Bext := gpu.NewConst("Bext", "T", mesh, mem, []float64{Bz, By, Bx}).Output()
	BeffBox.MAdd(Bext, 1)
	tBox.Alpha = 0.02
	solver.Advance(1e-9)

	for i := range avg {
		avg[i] = gpu.Sum(m.UnsafeData()[i].Device(), 0) / float32(mesh.NCell())
	}
	want = [3]float32{0.04370473, 0.11871325, -0.98528093}
	err = math.Sqrt(float64(sqr(avg[0]-want[0]) + sqr(avg[1]-want[1]) + sqr(avg[2]-want[2])))
	fmt.Println("avg:", avg, "err:", err)
	if err > 1e-2 {
		fmt.Println("FAILED")
		os.Exit(2)
	}
	fmt.Println("OK")
}

func sqr(x float32) float32 { return x * x }

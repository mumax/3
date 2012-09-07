package main

import (
	"fmt"
	"math"
	"nimble-cube/core"
	"nimble-cube/dump"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
	"os"
)

var (
	Msat, Aex, alpha, time        float64
	m, Hd, Hex, Heff, torque      [3][][][]float32
	m_, Hd_, Hex_, Heff_, torque_ [3][]float32
	demag                         conv.Conv
	size, pbc                     [3]int
	cellsize                      [3]float64
	step, out                     int
	B1, B2                        float32
)

func main() {
	N0, N1, N2 := 1, 32, 128
	size = [3]int{N0, N1, N2}
	cellsize = [3]float64{3e-9, 3.125e-9, 3.125e-9}

	// demag
	// TODO: use Msat (now it should be 1.0053, not 1)
	acc := 8
	pbc = [3]int{0, 0, 0}
	kernel := mag.BruteKernel(core.PadSize(size, pbc), cellsize, pbc, acc)
	demag = conv.NewBasic(size, kernel)

	m = demag.Input()
	m_ = core.Contiguous3(m)
	Hd = demag.Output()
	Hd_ = core.Contiguous3(Hd)

	theta := math.Pi / 8
	c := float32(math.Cos(theta))
	s := float32(math.Sin(theta))
	mag.SetAll(m, mag.Uniform(0, s, c))

	Hex = core.MakeVectors(size)
	Hex_ = core.Contiguous3(Hex)
	const mu0 = 4 * math.Pi * 1e-7
	Msat = 1.0053
	Aex = mu0 * 13e-12 / Msat

	Heff = core.MakeVectors(size)
	Heff_ = core.Contiguous3(Heff)

	alpha = 1
	torque = core.MakeVectors(size)
	torque_ = core.Contiguous3(torque)

	core.Log("relaxing")
	N := 10000
	dt := 100e-15
	for step = 0; step < N; step++ {
		time = dt * float64(step)
		update()
		mag.EulerStep(m_, torque_, dt)
	}

	alpha = 0.02
	time = 0

	B1, B2 = 4.3E-3, -24.6E-3
	core.Log("running")
	dt = 10e-15
	N = int(1e-9 / dt)
	for step = 0; step < N; step++ {
		time = dt * float64(step)
		update()
		if step%1000 == 0 {
			output()
		}
		mag.EulerStep(m_, torque_, dt)
	}
}

func update() {
	demag.Exec()
	mag.Exchange6(m, Hex, cellsize, Aex)
	core.Add3(Heff_, Hex_, Hd_)
	core.AddConst(Heff_[1], B1)
	core.AddConst(Heff_[2], B2)
	mag.LLGTorque(torque_, m_, Heff_, float32(alpha))
}

var (
	tablef *os.File
	table  dump.TableWriter
)

func output() {

	if tablef == nil {
		tablef = core.OpenFile("stdprobl4/m.table")
		table = dump.NewTableWriter(tablef, []string{"t", "mx", "my", "mz"}, []string{"s", "", "", ""})
	}

	dump.Quick(fmt.Sprintf("stdprobl4/m%06d", out), m[:])
	//	dump.Quick(fmt.Sprintf("t%06d", step), torque[:])
	//	dump.Quick(fmt.Sprintf("heff%06d", step), Heff[:])
	//	dump.Quick(fmt.Sprintf("hex%06d", step), Hex[:])
	//	dump.Quick(fmt.Sprintf("hd%06d", step), Hd[:])

	table.Data[0] = float32(time)
	table.Data[1] = float32(core.Average(m_[0]))
	table.Data[2] = float32(core.Average(m_[1]))
	table.Data[3] = float32(core.Average(m_[2]))
	table.WriteData()
	table.Flush()
	out++
}

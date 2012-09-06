package main

import (
	"flag"
	"fmt"
	"math"
	"nimble-cube/core"
	"nimble-cube/dump"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
	"strconv"
)

func main() {
	N0, N1, N2 := 1, 32, 128
	size := [3]int{N0, N1, N2}
	cellsize := [3]float64{3e-9, 3.125e-9, 3.125e-9}

	// demag
	// TODO: use Msat (now it should be 1.0053, not 1)
	acc := 8
	noPBC := [3]int{0, 0, 0}
	kernel := mag.BruteKernel(core.PadSize(size, noPBC), cellsize, noPBC, acc)
	demag := conv.NewBasic(size, kernel)

	m := demag.Input()
	m_ := core.Contiguous3(m)
	Hd := demag.Output()
	Hd_ := core.Contiguous3(Hd)

	theta := math.Pi / 8
	c := float32(math.Cos(theta))
	s := float32(math.Sin(theta))
	mag.SetAll(m, mag.Uniform(0, s, c))

	Hex := core.MakeVectors(size)
	Hex_ := core.Contiguous3(Hex)
	const mu0 = 4 * math.Pi * 1e-7
	Msat := 1.0053
	Aex := mu0 * 13e-12 / Msat

	Heff := core.MakeVectors(size)
	Heff_ := core.Contiguous3(Heff)

	alpha := float32(1)
	torque := core.MakeVectors(size)
	torque_ := core.Contiguous3(torque)

	out := core.OpenFile("m.table")
	defer out.Close()
	table := dump.NewTableWriter(out, []string{"t", "mx", "my", "mz"}, []string{"s", "", "", ""})
	defer table.Flush()

	N := 100000
	dt := 100e-15
	time := 0.
	for step := 0; step < N; step++ {
		time = dt * float64(step)

		demag.Exec()
		mag.Exchange6(m, Hex, cellsize, Aex)
		core.Add3(Heff_, Hex_, Hd_)
		mag.LLGTorque(torque_, m_, Heff_, alpha)

		if step%500 == 0 {
			dump.Quick(fmt.Sprintf("m%06d", step), m[:])
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
		}

		mag.EulerStep(m_, torque_, dt)
	}
}

func intflag(idx int) int {
	val, err := strconv.Atoi(flag.Arg(idx))
	core.Fatal(err)
	return val
}

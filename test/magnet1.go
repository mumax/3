package main

import (
	"flag"
	"math"
	"nimble-cube/core"
	"nimble-cube/dump"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
	"os"
	"strconv"
)

func main() {
	flag.Parse()
	N0, N1, N2 := intflag(0), intflag(1), intflag(2)
	//N := N0 * N1 * N2
	size := [3]int{N0, N1, N2}
	cellsize := [3]float64{3e-9, 3.125e-9, 3.125e-9}

	// demag
	acc := 4
	NOPBC := [3]int{0, 0, 0}
	// TODO: kernel should known its size depending on PBC or not!!
	kernel := mag.BruteKernel(core.PadSize(size, NOPBC), cellsize, NOPBC, acc)
	demag := conv.NewBasic(size, kernel)
	m := demag.Input()
	m_ := core.Contiguous3(m)
	Hd := demag.Output()
	Hd_ := core.Contiguous3(Hd)

	// inital mag
	y1, y2 := 0,N1//3*N1/8, 5*N1/8
	z1, z2 := 0,N2//0*N2/8, 2*N2/8
	my := m[1]
	mz := m[2]
	theta := 0.01
	c := float32(math.Cos(theta))
	s := float32(math.Sin(theta))
	for i := range mz {
		for j := y1; j < y2; j++ {
			for k := z1; k < z2; k++ {
				mz[i][j][k] = c
				my[i][j][k] = s
			}
		}
	}

	dump.Quick("m", m[:])

	demag.Exec()
	dump.Quick("hd", Hd[:])

	Hex := core.MakeVectors(size)
	Hex_ := core.Contiguous3(Hex)
	Aex := 13e-12
	mag.Exchange6(m, Hex, cellsize, Aex)
	dump.Quick("hex", Hex[:])

	Heff := core.MakeVectors(size)
	Heff_ := core.Contiguous3(Heff)
	core.Add3(Heff_, Hex_, Hd_)
	dump.Quick("heff", Heff[:])

	alpha := float32(1)
	torque := core.MakeVectors(size)
	torque_ := core.Contiguous3(torque)
	mag.LLGTorque(torque_, m_, Heff_, alpha)
	dump.Quick("torque", torque[:])

	out, err := os.OpenFile("m.table", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	core.Fatal(err)
	defer out.Close()
	table := dump.NewTableWriter(out, []string{"t", "mx", "my", "mz"}, []string{"s", "", "", ""})
	defer table.Flush()

	N := 1000
	dt := 10e-15
	time := 0.
	for step := 0; step < N; step++ {
		time = dt * float64(step)
		demag.Exec()
		mag.Exchange6(m, Hex, cellsize, Aex)
		core.Add3(Heff_, Hex_, Hd_)
		mag.LLGTorque(torque_, m_, Heff_, alpha)
		mag.EulerStep(m_, torque_, dt)
		//dump.Quick(fmt.Sprintf("m%06d", step), m[:])
		table.Data[0] = float32(time)
		table.Data[1] = float32(core.Average(m_[0]))
		table.Data[2] = float32(core.Average(m_[1]))
		table.Data[3] = float32(core.Average(m_[2]))
		table.WriteData()
	}
}

func intflag(idx int) int {
	val, err := strconv.Atoi(flag.Arg(idx))
	core.Fatal(err)
	return val
}

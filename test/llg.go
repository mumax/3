package main

import (
	"fmt"
	"nimble-cube/core"
	"nimble-cube/dump"
	"nimble-cube/mag"
	"os"
)

func main() {
	N0, N1, N2 := 1, 1, 1
	size := [3]int{N0, N1, N2}
	//cellsize := [3]float64{1e-9, 1e-9, 1e-9}

	m := core.MakeVectors(size)
	h := core.MakeVectors(size)
	torque := core.MakeVectors(size)

	m_ := core.Contiguous3(m)
	h_ := core.Contiguous3(h)
	torque_ := core.Contiguous3(torque)
	alpha := float32(0.01)

	mag.SetAll(m, mag.Uniform(0, 0, 1))
	mag.SetAll(h, mag.Uniform(1, 0, 0))
	mag.LLGTorque(torque_, m_, h_, alpha)
	fmt.Println(m_[0][0], m_[1][0], m_[2][0], h_[0][0], h_[1][0], h_[2][0], torque_[0][0], torque_[1][0], torque_[2][0])

	out, err := os.OpenFile("m.table", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	core.Fatal(err)
	defer out.Close()
	table := dump.NewTableWriter(out, []string{"t", "mx", "my", "mz"}, []string{"s", "", "", ""})
	defer table.Flush()

	N := 100000
	dt := 10e-15
	time := 0.
	for step := 0; step < N; step++ {
		time = dt * float64(step)
		mag.LLGTorque(torque_, m_, h_, alpha)
		mag.EulerStep(m_, torque_, dt)
		if step%100 == 0 {
			table.Data[0] = float32(time)
			table.Data[1] = float32(core.Average(m_[0]))
			table.Data[2] = float32(core.Average(m_[1]))
			table.Data[3] = float32(core.Average(m_[2]))
			table.WriteData()
		}
	}
}

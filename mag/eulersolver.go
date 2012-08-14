package mag

import (
	"nimble-cube/core"
)

// Euler solver.
type EulerBox struct {
	size [3]int
	n int
	M              [3][]float32
	Torque         [3][]float32
	Time           float64
	Dt             float64
	Step           int
	numSlice, sliceLen int
}

func NewEuler(size [3]int, numSlice)*EulerBox{
	e := new(EulerBox)
	e.size = size
	e.n = core.Prod(size)
	e.numSlice = numSlice
	core.Assert(n%numSlice == 0)
	e.sliceLen = n/numSlice	
	return e
}

func (box *EulerBox) Run(steps int) {

	for s := 0; s < steps; s++ {

		for w := 0; w < box.nWarp; w++ {
			start := w * box.warpLen
			stop := (w + 1) * box.warpLen
			for i := start; i < stop; i++ {

				var m1 Vector
				m1[X] = box.M[X][i] + float32(box.Dt)*box.Torque[X][i]
				m1[Y] = box.M[Y][i] + float32(box.Dt)*box.Torque[Y][i]
				m1[Z] = box.M[Z][i] + float32(box.Dt)*box.Torque[Z][i]
				m1 = m1.Normalized()
				box.M[X][i] = m1[X]
				box.M[Y][i] = m1[Y]
				box.M[Z][i] = m1[Z]

			}
		}
		box.Time += (box.Dt)
		box.Step++
	}
}

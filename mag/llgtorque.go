package mag

import (
	//"nimble-cube/core"
	"nimble-cube/gpu/conv"
)

// Landau-Lifshitz torque.
type LLGBox struct {
	nWarp, warpLen int
	M              [3][]float32
	H              [3][]float32
	alpha          []float32
	Torque         [3][]float32
	hplan          *conv.Symmetric
	//solver *Euler
}

func (box *LLGBox) Run() {

	for {
		for w := 0; w < box.nWarp; w++ {
			start := w * box.warpLen
			stop := (w + 1) * box.warpLen

			box.hplan.Pull(stop)

			for i := start; i < stop; i++ {

				var m Vector
				var h Vector
				m[X], m[Y], m[Z] = box.M[X][i], box.M[Y][i], box.M[Z][i]
				h[X], h[Y], h[Z] = box.H[X][i], box.H[Y][i], box.H[Z][i]

				alpha := box.alpha[i]

				mxh := m.Cross(h)
				t := mxh.Sub(m.Cross(mxh).Scaled(alpha))

				box.Torque[X][i] = t[X]
				box.Torque[Y][i] = t[Y]
				box.Torque[Z][i] = t[Z]
			}

			//solver.
		}
	}
}

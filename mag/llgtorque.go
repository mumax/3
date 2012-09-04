package mag

import (
)

func LLGTorque(torque, m, H [3][]float32, alpha float32) {
	//const mu0 = 4 * math.Pi * 1e7
	const gamma =1 //= 1.76085970839e11 / mu0 // rad/Ts
	for i := range torque[0] {

		var m_ Vector
		var H_ Vector
		m_[X], m_[Y], m_[Z] = m[X][i], m[Y][i], m[Z][i]
		H_[X], H_[Y], H_[Z] = H[X][i], H[Y][i], H[Z][i]

		mxh := m_.Cross(H_) // not inlined for some reason...
		t_ := (mxh.Sub(m_.Cross(mxh).Scaled(alpha))).Scaled(gamma)

		torque[X][i] = t_[X]
		torque[Y][i] = t_[Y]
		torque[Z][i] = t_[Z]
	}
}

// Landau-Lifshitz torque.
type LLGBox struct {
	nWarp, warpLen int
	M              [3][]float32
	H              [3][]float32
	alpha          []float32
	Torque         [3][]float32
	//hplan          *conv.Symmetric
	//solver *Euler
}

func (box *LLGBox) Run() {

	for {
		for w := 0; w < box.nWarp; w++ {
			start := w * box.warpLen
			stop := (w + 1) * box.warpLen

			//box.hplan.Pull(stop)

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

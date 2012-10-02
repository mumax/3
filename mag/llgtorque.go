package mag

import "nimble-cube/core"

type LLGTorque struct {
	torque core.Chan3
	m, h   core.RChan3
	alpha  float32
}

func NewLLGTorque(torque core.Chan3, m, h core.RChan3, alpha float32) *LLGTorque {
	core.Assert(torque.Size() == m.Size())
	core.Assert(torque.Size() == h.Size())
	return &LLGTorque{torque, m, h, alpha}
}

func (r *LLGTorque) Run() {
	n := core.BlockLen(r.torque.Size())
	for {
		M := r.m.ReadNext(n)
		H := r.h.ReadNext(n)
		T := r.torque.WriteNext(n)
		llgTorque(T, M, H, r.alpha)
		r.torque.WriteDone()
		r.m.ReadDone()
		r.h.ReadDone()
	}
}

func llgTorque(torque, m, H [3][]float32, alpha float32) {
	const gamma = 1.76085970839e11 // rad/Ts // TODO

	var mx, my, mz float32
	var hx, hy, hz float32

	for i := range torque[0] {
		mx, my, mz = m[X][i], m[Y][i], m[Z][i]
		hx, hy, hz = H[X][i], H[Y][i], H[Z][i]

		mxhx := my*hz - mz*hy
		mxhy := -mx*hz + mz*hx
		mxhz := mx*hy - my*hx

		mxmxhx := my*mxhz - mz*mxhy
		mxmxhy := -mx*mxhz + mz*mxhx
		mxmxhz := mx*mxhy - my*mxhx

		torque[X][i] = gamma * (mxhx - alpha*mxmxhx) // todo: gilbert factor
		torque[Y][i] = gamma * (mxhy - alpha*mxmxhy)
		torque[Z][i] = gamma * (mxhz - alpha*mxmxhz)
	}
}

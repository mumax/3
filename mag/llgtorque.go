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
	// TODO: properly block
	for {
		T := r.torque.WriteNext(n)
		M := r.m.ReadNext(n)
		H := r.h.ReadNext(n)
		llgTorque(T, M, H, r.alpha)
		r.torque.WriteDone()
		r.m.ReadDone()
		r.h.ReadDone()
	}
}

func llgTorque(torque, m, H [3][]float32, alpha float32) {
	const gamma = 1.76085970839e11 // rad/Ts // TODO
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

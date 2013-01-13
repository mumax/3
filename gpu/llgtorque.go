package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/gpu/ptx"
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/safe"
)

// LLGTorque calculates the REDUCED torque:
// 	- m x B +  α m x (m x B)
type LLGTorque struct {
	torque nimble.ChanN
	m, b   nimble.RChanN
	Alpha  float32
}

func NewLLGTorque(tag string, m_, B_ nimble.ChanN, alpha float32) *LLGTorque {
	m, B := m_.NewReader(), B_.NewReader()
	core.Assert(B.Size() == m.Size())
	torque := nimble.MakeChanN(3, tag, "T", m.Mesh(), m_.MemType(), 1)
	tq := &LLGTorque{torque, m, B, alpha}
	nimble.Stack(tq)
	return tq
}

func (r *LLGTorque) Output() nimble.ChanN { return r.torque }

func (r *LLGTorque) Run() {
	LockCudaThread()
	for {
		r.Exec()
	}
}

func (r *LLGTorque) Exec() {
	n := r.torque.Mesh().NCell() // TODO: blocksize
	M := Device3(r.m.ReadNext(n))
	B := Device3(r.b.ReadNext(n))
	T := Device3(r.torque.WriteNext(n))

	CalcLLGTorque(T, M, B, r.Alpha)

	r.torque.WriteDone()
	r.m.ReadDone()
	r.b.ReadDone()
}

func CalcLLGTorque(torque, m, B [3]safe.Float32s, alpha float32) {
	N := torque[0].Len()
	gridDim, blockDim := Make1DConf(N)
	ptx.K_llgtorque(torque[0].Pointer(), torque[1].Pointer(), torque[2].Pointer(),
		m[0].Pointer(), m[1].Pointer(), m[2].Pointer(),
		B[0].Pointer(), B[1].Pointer(), B[2].Pointer(),
		alpha, N, gridDim, blockDim)
}

func Device3(s []nimble.Slice) [3]safe.Float32s {
	core.Assert(len(s) == 3)
	return [3]safe.Float32s{s[0].Device(), s[1].Device(), s[2].Device()}
}

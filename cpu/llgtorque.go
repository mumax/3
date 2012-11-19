package cpu

import (
	"code.google.com/p/nimble-cube/nimble"
	"code.google.com/p/nimble-cube/core"
)

// LLG Torque / gamma
type LLGTorque struct {
	torque nimble.Chan3
	m, b   nimble.RChan3
	alpha  float32
	bExt   Vector
}

func NewLLGTorque(torque nimble.Chan3, m, B nimble.RChan3, alpha float32) *LLGTorque {
	core.Assert(torque.Mesh().Size() == m.Mesh().Size())
	core.Assert(torque.Mesh().Size() == B.Mesh().Size())
	return &LLGTorque{torque, m, B, alpha, Vector{0, 0, 0}}
}

func (r *LLGTorque) Run() {
	n := r.torque.ChanN().BufLen()
	for {
		M := Host3(r.m.ReadNext(n))
		B := Host3(r.b.ReadNext(n))
		T := Host3(r.torque.WriteNext(n))
		llgTorque(T, M, B, r.alpha, r.bExt)
		r.torque.WriteDone()
		r.m.ReadDone()
		r.b.ReadDone()
	}
}

func llgTorque(torque, m, B [3][]float32, alpha float32, bExt Vector) {
	//const gamma = 1.76085970839e11 // rad/Ts // TODO

	var mx, my, mz float32
	var Bx, By, Bz float32

	for i := range torque[0] {
		mx, my, mz = m[X][i], m[Y][i], m[Z][i]
		Bx = B[X][i] + bExt[X]
		By = B[Y][i] + bExt[Y]
		Bz = B[Z][i] + bExt[Z]

		mxBx := my*Bz - mz*By
		mxBy := -mx*Bz + mz*Bx
		mxBz := mx*By - my*Bx

		mxmxBx := my*mxBz - mz*mxBy
		mxmxBy := -mx*mxBz + mz*mxBx
		mxmxBz := mx*mxBy - my*mxBx

		torque[X][i] = (mxBx - alpha*mxmxBx) // todo: gilbert factor
		torque[Y][i] = (mxBy - alpha*mxmxBy)
		torque[Z][i] = (mxBz - alpha*mxmxBz)
	}
}

func Host3(s [3]nimble.Slice)[3][]float32{
	return [3][]float32{s[0].Host(), s[1].Host(), s[2].Host() }
}

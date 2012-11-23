package cpu

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/nimble"
)

// LLG Torque / gamma
type LLGTorque struct {
	torque nimble.ChanN
	m, b   nimble.RChanN
	alpha  float32
	bExt   Vector
}

func NewLLGTorque(tag string, m_, B_ nimble.ChanN, alpha float32) *LLGTorque {
	m := m_.NewReader()
	B := B_.NewReader()
	core.Assert(B.Mesh().Size() == m.Mesh().Size())
	torque := nimble.MakeChanN(3, tag, "T", m.Mesh(), m_.MemType(), 1)
	return &LLGTorque{torque, m, B, alpha, Vector{0, 0, 0}}
}

func (r *LLGTorque) Output() nimble.ChanN { return r.torque }


func (r *LLGTorque) Run() {
	n := r.torque.BufLen()
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

func Host3(s []nimble.Slice) [3][]float32 {
	core.Assert(len(s) == 3)
	return [3][]float32{s[0].Host(), s[1].Host(), s[2].Host()}
}

func Host(s []nimble.Slice) [][]float32 {
	h := make([][]float32, len(s))
	for i := range h {
		h[i] = s[i].Host()
	}
	return h
}

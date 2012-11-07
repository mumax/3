package mag

//import "nimble-cube/core"
//
//// LLG Torque / gamma
//type LLGTorque struct {
//	torque core.Chan3
//	m, b   core.RChan3
//	alpha  float32
//	bExt   Vector
//	Func   func(t, m, B [3][]float32, a float32, b Vector)
//}
//
//func NewLLGTorque(torque core.Chan3, m, B core.RChan3, alpha float32) *LLGTorque {
//	core.Assert(torque.Mesh().Size() == m.Mesh().Size())
//	core.Assert(torque.Mesh().Size() == B.Mesh().Size())
//	return &LLGTorque{torque, m, B, alpha, Vector{0, 0, 0}, llgTorque}
//}
//
//func (r *LLGTorque) Run() {
//	n := core.BlockLen(r.torque.Mesh().Size())
//	for {
//		M := r.m.ReadNext(n)
//		B := r.b.ReadNext(n)
//		T := r.torque.WriteNext(n)
//		r.Func(T, M, B, r.alpha, r.bExt)
//		r.torque.WriteDone()
//		r.m.ReadDone()
//		r.b.ReadDone()
//	}
//}
//
//func llgTorque(torque, m, B [3][]float32, alpha float32, bExt Vector) {
//	//const gamma = 1.76085970839e11 // rad/Ts // TODO
//
//	var mx, my, mz float32
//	var Bx, By, Bz float32
//
//	for i := range torque[0] {
//		mx, my, mz = m[X][i], m[Y][i], m[Z][i]
//		Bx = B[X][i] + bExt[X]
//		By = B[Y][i] + bExt[Y]
//		Bz = B[Z][i] + bExt[Z]
//
//		mxBx := my*Bz - mz*By
//		mxBy := -mx*Bz + mz*Bx
//		mxBz := mx*By - my*Bx
//
//		mxmxBx := my*mxBz - mz*mxBy
//		mxmxBy := -mx*mxBz + mz*mxBx
//		mxmxBz := mx*mxBy - my*mxBx
//
//		torque[X][i] = (mxBx - alpha*mxmxBx) // todo: gilbert factor
//		torque[Y][i] = (mxBy - alpha*mxmxBy)
//		torque[Z][i] = (mxBz - alpha*mxmxBz)
//	}
//}

package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"unsafe"
)

// LLGTorque calculates the REDUCED torque:
// 	- m x B +  α m x (m x B)
type LLGTorque struct {
	torque nimble.ChanN
	m, b   nimble.RChanN
	Alpha  float32
	bExt   [3]float32
	stream cu.Stream
}

func NewLLGTorque(tag string, m_, B_ nimble.ChanN, alpha float32) *LLGTorque {
	m, B := m_.NewReader(), B_.NewReader()
	core.Assert(B.Size() == m.Size())
	torque := nimble.MakeChanN(3, tag, "T", m.Mesh(), m_.MemType(), 1)
	tq := &LLGTorque{torque, m, B, alpha, [3]float32{0, 0, 0}, cu.StreamCreate()}
	nimble.Stack(tq)
	return tq
}

func (r *LLGTorque) Output() nimble.ChanN { return r.torque }

func (r *LLGTorque) Run() {
	LockCudaThread()
	for {
		r.Exec() // TODO: exec(t,m,b)
	}
}

func (r *LLGTorque) Exec() {
	n := r.torque.Mesh().NCell() // TODO: blocksize
	M := Device3(r.m.ReadNext(n))
	B := Device3(r.b.ReadNext(n))
	T := Device3(r.torque.WriteNext(n))

	CalcLLGTorque(T, M, B, r.Alpha, r.bExt, r.stream)

	r.torque.WriteDone()
	r.m.ReadDone()
	r.b.ReadDone()
}

func CalcLLGTorque(torque, m, B [3]safe.Float32s, alpha float32, bExt [3]float32, stream cu.Stream) {
	core.Assert(bExt == [3]float32{0, 0, 0})

	llgtorqueCode := PTXLoad("llgtorque")

	N := torque[0].Len()
	gridDim, blockDim := Make1DConf(N)

	t0ptr := torque[0].Pointer()
	t1ptr := torque[1].Pointer()
	t2ptr := torque[2].Pointer()
	m0ptr := m[0].Pointer()
	m1ptr := m[1].Pointer()
	m2ptr := m[2].Pointer()
	B0ptr := B[0].Pointer()
	B1ptr := B[1].Pointer()
	B2ptr := B[2].Pointer()

	args := []unsafe.Pointer{
		unsafe.Pointer(&t0ptr),
		unsafe.Pointer(&t1ptr),
		unsafe.Pointer(&t2ptr),
		unsafe.Pointer(&m0ptr),
		unsafe.Pointer(&m1ptr),
		unsafe.Pointer(&m2ptr),
		unsafe.Pointer(&B0ptr),
		unsafe.Pointer(&B1ptr),
		unsafe.Pointer(&B2ptr),
		unsafe.Pointer(&alpha),
		unsafe.Pointer(&N)}

	shmem := 0
	cu.LaunchKernel(llgtorqueCode, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
	stream.Synchronize()
}

func Device3(s []nimble.Slice) [3]safe.Float32s {
	core.Assert(len(s) == 3)
	return [3]safe.Float32s{s[0].Device(), s[1].Device(), s[2].Device()}
}

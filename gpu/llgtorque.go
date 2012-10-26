package gpu

import (
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"nimble-cube/core"
	"nimble-cube/gpu/ptx"
	"nimble-cube/mag"
	"unsafe"
)

type LLGTorque struct {
	torque Chan3
	m, b   RChan3
	alpha  float32
	bExt   mag.Vector
	stream cu.Stream
}

func RunLLGTorque(tag string, m, B RChan3, alpha float32) *LLGTorque {
	core.Assert(B.Size() == m.Size())
	torque := MakeChan3(tag, "T", m.Mesh(), 1)
	tq := &LLGTorque{torque, m, B, alpha, mag.Vector{0, 0, 0}, cu.StreamCreate()}
	core.Stack(tq)
	return tq
}

func (r *LLGTorque) Output() Chan3 { return r.torque }

func (r *LLGTorque) Run() {
	LockCudaThread()
	for {
		r.Exec()
	}
}

func (r *LLGTorque) Exec() {
	n := core.Prod(r.torque.Size())
	M := r.m.ReadNext(n)
	B := r.b.ReadNext(n)
	T := r.torque.WriteNext(n)

	llgtorque(T, M, B, r.alpha, r.bExt, r.stream)

	r.torque.WriteDone()
	r.m.ReadDone()
	r.b.ReadDone()
}

var llgtorqueCode cu.Function

func llgtorque(torque, m, B [3]safe.Float32s, alpha float32, bExt mag.Vector, stream cu.Stream) {
	core.Assert(bExt == mag.Vector{0, 0, 0})

	if llgtorqueCode == 0 {
		mod := cu.ModuleLoadData(ptx.LLGTORQUE)
		llgtorqueCode = mod.GetFunction("llgtorque")
	}

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

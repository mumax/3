package gpu

import (
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"nimble-cube/core"
	"nimble-cube/gpu/ptx"
	"unsafe"
)

type Adder3 struct {
	sum          Chan3
	term1, term2 RChan3
	fac1, fac2   float32
	stream       cu.Stream
}

func NewAdder3(sum Chan3, term1 RChan3, factor1 float32, term2 RChan3, factor2 float32) *Adder3 {
	core.Assert(sum.Size() == term1.Size())
	core.Assert(sum.Size() == term2.Size())
	return &Adder3{sum, term1, term2, factor1, factor2, cu.StreamCreate()}
}

func (a *Adder3) Run() {
	LockCudaThread()
	for {
		a.Exec()
	}
}

func (a *Adder3) Exec() {
	N := core.Prod(a.sum.Size())

	a.term1.ReadNext(N)
	a.term2.ReadNext(N)
	a.sum.WriteNext(N)

	for i := 0; i < 3; i++ {
		madd(a.sum.UnsafeData()[i], a.term1.UnsafeData()[i], a.fac1, a.term2.UnsafeData()[i], a.fac2, a.stream)
	}

	a.sum.WriteDone()
	a.term1.ReadDone()
	a.term2.ReadDone()
}

var maddCode cu.Function

func madd(dst, src1 safe.Float32s, factor1 float32, src2 safe.Float32s, factor2 float32, stream cu.Stream) {

	core.Assert(dst.Len() == src1.Len() && dst.Len() == src2.Len())

	if maddCode == 0 {
		mod := cu.ModuleLoadData(ptx.MADD)
		maddCode = mod.GetFunction("madd")
	}

	N := dst.Len()
	gridDim, blockDim := Make1DConf(N)

	dstptr := dst.Pointer()
	src1ptr := src1.Pointer()
	src2ptr := src2.Pointer()

	args := []unsafe.Pointer{
		unsafe.Pointer(&dstptr),
		unsafe.Pointer(&src1ptr),
		unsafe.Pointer(&factor1),
		unsafe.Pointer(&src2ptr),
		unsafe.Pointer(&factor2),
		unsafe.Pointer(&N)}

	shmem := 0
	cu.LaunchKernel(maddCode, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
	stream.Synchronize()
}

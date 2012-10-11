package gpu

import (
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"nimble-cube/core"
	"nimble-cube/gpu/ptx"
	"unsafe"
)

type Adder3 struct {
	sum       Chan3
	terms     []RChan3
	blocksize int
}

func NewAdder3(sum Chan3, terms ...RChan3) *Adder3 {
	core.Assert(len(terms) > 1)
	for _, t := range terms {
		core.Assert(t.Size() == sum.Size())
	}
	return &Adder3{sum, terms, core.BlockLen(sum.Size())}
}

func (a *Adder3) Run() {

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

package gpu

import (
	"github.com/barnex/cuda5/cu"
	//	"github.com/barnex/cuda5/safe"
	"fmt"
	"nimble-cube/core"

//	"nimble-cube/gpu/ptx"
//	"unsafe"
)

type Sum struct {
	sum     ChanN
	term    []RChanN
	weight  []float32
	stream  cu.Stream
	running bool
}

func RunSum(tag string, term1_ Chan, weight1 float32, term2_ Chan, weight2 float32, nBlocks ...int) *Sum {
	term1, term2 := term1_.ChanN().NewReader(), term2_.ChanN().NewReader()
	output := MakeChanN(term1.NComp(), tag, term1.Unit(), term1.Mesh(), nBlocks...)
	sum := &Sum{sum: output, stream: cu.StreamCreate()}
	sum.MAdd(term1, weight1)
	sum.MAdd(term2, weight2)
	core.Stack(sum)
	return sum
}

func (s *Sum) MAdd(term RChanN, weight float32) {
	// Fail-fast check. May not always fire but should be OK to catch obvious bugs.
	if s.running {
		core.Fatal(fmt.Errorf("sum: madd: already running"))
	}

	if len(s.term) != 0 {
		core.Assert(term.Size() == s.term[0].Size())
		core.Assert(term.Unit() == s.term[0].Unit())   // TODO: nice error handling
		core.Assert(*term.Mesh() == *s.term[0].Mesh()) // TODO: nice error handling
	}
	s.term = append(s.term, term)
	s.weight = append(s.weight, weight)
}

func (s *Sum) Run() {
	s.running = true
	LockCudaThread()
	for {
		s.Exec()
	}
}

func (s *Sum) Exec() {
	N := s.sum.BlockLen()
	nComp := s.sum.NComp()

	// TODO: components could be streamed
	for c := 0; c < nComp; c++ {

		A := s.term[0][c].ReadNext(N)
		B := s.term[1][c].ReadNext(N)
		S := s.sum[c].WriteNext(N)

		madd(S, A, s.weight[0], B, s.weight[1], s.stream)

		s.term[0][c].ReadDone()
		s.term[1][c].ReadDone()

		for t := 2; t < len(s.term); t++ {
			C := s.term[t][c].ReadNext(N)
			madd(S, S, 1, C, s.weight[t], s.stream)
			s.term[t][c].ReadDone()
		}
		s.sum.WriteDone()
	}
}

func (s *Sum) Output() ChanN { return s.sum }

//var maddCode cu.Function
//
//func madd(dst, src1 safe.Float32s, factor1 float32, src2 safe.Float32s, factor2 float32, stream cu.Stream) {
//
//	core.Assert(dst.Len() == src1.Len() && dst.Len() == src2.Len())
//
//	if maddCode == 0 {
//		mod := cu.ModuleLoadData(ptx.MADD)
//		maddCode = mod.GetFunction("madd")
//	}
//
//	N := dst.Len()
//	gridDim, blockDim := Make1DConf(N)
//
//	dstptr := dst.Pointer()
//	src1ptr := src1.Pointer()
//	src2ptr := src2.Pointer()
//
//	args := []unsafe.Pointer{
//		unsafe.Pointer(&dstptr),
//		unsafe.Pointer(&src1ptr),
//		unsafe.Pointer(&factor1),
//		unsafe.Pointer(&src2ptr),
//		unsafe.Pointer(&factor2),
//		unsafe.Pointer(&N)}
//
//	shmem := 0
//	cu.LaunchKernel(maddCode, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
//	stream.Synchronize()
//}

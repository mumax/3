package gpu

import (
	"fmt"
	"github.com/barnex/cuda5/cu"
	"nimble-cube/core"
)

type Sum struct {
	sum     ChanN
	term    []RChanN
	weight  []float32
	stream  cu.Stream
	running bool 
	readlen int // smallest of all blockLen's.
}

func RunSum(tag string, term1_ Chan, weight1 float32, term2_ Chan, weight2 float32, nBlocks ...int) *Sum {
	term1, term2 := term1_.ChanN().NewReader(), term2_.ChanN().NewReader()
	output := MakeChanN(term1.NComp(), tag, term1.Unit(), term1.Mesh(), nBlocks...)
	sum := &Sum{sum: output, stream: cu.StreamCreate()}
	sum.MAdd(term1, weight1)
	sum.MAdd(term2, weight2)
	sum.readlen = output.BlockLen() // TODO: core.Min(output.BlockLen(), core.Min(term1.BlockLen(), term2.BlockLen()))
	core.Stack(sum)
	return sum
}

func (s *Sum) MAdd(term RChanN, weight float32) {
	// Fail-fast check. May not always fire but should be OK to catch obvious bugs.
	if s.running {
		core.Fatal(fmt.Errorf("sum: madd: already running"))
	}

	if len(s.term) != 0 {
		core.CheckEqualSize(term.Size(), s.term[0].Size())
		core.CheckUnits(term.Unit(), s.term[0].Unit())
		core.Assert(*term.Mesh() == *s.term[0].Mesh()) // TODO: nice error handling
	}
	s.term = append(s.term, term)
	s.weight = append(s.weight, weight)
	//TODO:
	//s.readlen = core.Min(s.readlen, term.BlockLen())
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
		s.sum[c].WriteDone()
	}
}

func (s *Sum) Output() ChanN { return s.sum }

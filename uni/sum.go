package uni

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/cu"
)

// Universal weighed sum.
type Sum struct {
	sum    nimble.ChanN
	term   []nimble.RChanN
	weight []float32
	stream cu.Stream
	dev    Device
}

func NewSum(tag string, term1, term2 nimble.Chan, weight1, weight2 float32, mem nimble.MemType, dev Device) *Sum {
	//t1, t2 := term1.ChanN().NewReader(), term2.ChanN().NewReader()
	output := nimble.MakeChanN(term1.NComp(), tag, term1.Unit(), term1.ChanN().Mesh(), mem, 1)
	sum := &Sum{sum: output, dev: dev, stream: dev.StreamCreate()}
	sum.MAdd(term1, weight1)
	sum.MAdd(term2, weight2)
	nimble.Stack(sum)
	return sum
}

func (s *Sum) MAdd(term_ nimble.Chan, weight float32) {
	term := term_.ChanN().NewReader()
	if len(s.term) != 0 {
		core.Assert(term.NComp() == s.sum.NComp())
		core.CheckEqualSize(term.Size(), s.term[0].Size())
		core.CheckUnits(term.Unit(), s.term[0].Unit())
		core.Assert(*term.Mesh() == *s.term[0].Mesh()) // TODO: nice error handling
	}
	s.term = append(s.term, term)
	s.weight = append(s.weight, weight)
}

func (s *Sum) Run() {
	s.dev.InitThread()
	for {
		s.Exec()
	}
}

func (s *Sum) Exec() {
	N := s.sum.BufLen()
	nComp := s.sum.NComp()

	// TODO: components could be streamed in parallel.
	for c := 0; c < nComp; c++ {

		A := s.term[0].Comp(c).ReadNext(N)
		B := s.term[1].Comp(c).ReadNext(N)
		S := s.sum.Comp(c).WriteNext(N)

		s.dev.Madd(S, A, B, s.weight[0], s.weight[1], s.stream)

		s.term[0].Comp(c).ReadDone()
		s.term[1].Comp(c).ReadDone()

		for t := 2; t < len(s.term); t++ {
			C := s.term[t].Comp(c).ReadNext(N)
			s.dev.Madd(S, S, C, 1, s.weight[t], s.stream)
			s.term[t].Comp(c).ReadDone()
		}
		s.sum.Comp(c).WriteDone()
	}
	if s.stream != 0 {
		s.stream.Synchronize()
	}
}

func (s *Sum) Output() nimble.ChanN { return s.sum }

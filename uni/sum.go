package uni

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
)

// Universal weighed sum.
type Sum struct {
	sum    nimble.ChanN
	term   []nimble.RChanN
	weight []float32
	dev    Device
}

func NewSum(tag string, term1, term2 nimble.ChanN, weight1, weight2 float32, mem nimble.MemType, dev Device) *Sum {
	output := nimble.MakeChanN(term1.NComp(), tag, term1.Unit(), term1.Mesh(), mem, 1)
	sum := &Sum{sum: output, dev: dev}
	sum.MAdd(term1, weight1)
	sum.MAdd(term2, weight2)
	nimble.Stack(sum)
	return sum
}

// Add term with weight 1.
func (s *Sum) Add(term_ nimble.ChanN) {
	s.MAdd(term_, 1)
}

// Add term * weight to the sum.
// TODO: it might be nice to add to separate components
func (s *Sum) MAdd(term_ nimble.ChanN, weight float32) {
	term := term_.NewReader()
	if len(s.term) != 0 {
		core.Assert(term.NComp() == s.sum.NComp())
		core.CheckEqualSize(term.Mesh().Size(), s.term[0].Mesh().Size())
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

		Ac := s.term[0].Comp(c)
		Bc := s.term[1].Comp(c)
		Sc := s.sum.Comp(c)
		S := Sc.WriteNext(N)
		A := Ac.ReadNext(N)
		B := Bc.ReadNext(N)

		s.dev.Madd(S, A, B, s.weight[0], s.weight[1])

		Ac.ReadDone()
		Bc.ReadDone()

		for t := 2; t < len(s.term); t++ {
			Cc := s.term[t].Comp(c)
			C := Cc.ReadNext(N)
			s.dev.Madd(S, S, C, 1, s.weight[t])
			Cc.ReadDone()
		}
		Sc.WriteDone()
	}
}

func (s *Sum) Output() nimble.ChanN { return s.sum }

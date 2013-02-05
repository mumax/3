package gpu

import (
	"code.google.com/p/mx3/nimble"
)

type Stencil3D struct {
	in     nimble.RChanN
	out    nimble.ChanN
	Weight [3][3][7]float32
}

// Stencil adds the stencil result to out.
// Uses the same MemType as in.
func NewStencil3D(tag, unit string, in nimble.ChanN) *Stencil3D {
	r := in.NewReader() // TODO: buffer
	w := nimble.MakeChanN(3, tag, unit, r.Mesh(), in.MemType(), -1)
	return &Stencil3D{in: r, out: w} // Weight inits to zeros.
}

func (s *Stencil3D) Exec() {
	out := s.out.UnsafeData()
	in := s.in.UnsafeData()
	s.exec(out, in)
}

func (s *Stencil3D) Run() {
	N := s.out.Mesh().NCell()
	LockCudaThread()
	for {
		in := s.in.ReadNext(N)
		out := s.out.WriteNext(N)
		s.exec(out, in)
		s.in.ReadDone()
		s.out.WriteDone()
	}
}

func (s *Stencil3D) exec(out, in nimble.Slice) {
	m := s.out.Mesh()
	Memset(out, 0)
	for di := 0; di < 3; di++ {
		dst := out.Comp(di)
		for si := 0; si < 3; si++ {
			src := in.Comp(si)
			StencilAdd(dst, src, m, &(s.Weight[di][si]))
		}
	}
}

func (s *Stencil3D) Output() nimble.ChanN {
	return s.out
}

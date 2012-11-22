package gpu

import (
	"code.google.com/p/nimble-cube/nimble"
	"github.com/barnex/cuda5/cu"
)

type Stencil3D struct {
	in     nimble.RChanN
	out    nimble.ChanN
	Weight [3][3][7]float32
	stream cu.Stream
}

// Stencil adds the stencil result to out.
// Uses the same MemType as in.
func NewStencil3D(tag, unit string, in nimble.ChanN) *Stencil3D {
	r := in.NewReader() // TODO: buffer
	w := nimble.MakeChanN(3, tag, unit, r.Mesh(), in.MemType(), -1)
	return &Stencil3D{in: r, out: w, stream: cu.StreamCreate()}
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

func (s *Stencil3D) exec(out, in []nimble.Slice) {
	m := s.out.Mesh()
	for di := range out {
		dst := out[di].Device()
		dst.Memset(0)
		for si := range in {
			src := in[si].Device()
			StencilAdd(dst, src, m, &(s.Weight[di][si]), s.stream)
		}
	}
}

func (s *Stencil3D) Output() nimble.ChanN {
	return s.out
}

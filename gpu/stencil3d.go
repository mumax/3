package gpu

import (
	//"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/nimble"
	"github.com/barnex/cuda5/cu"
	//"github.com/barnex/cuda5/safe"
	//"unsafe"
)

type Stencil3D struct {
	in     nimble.RChanN
	out    nimble.ChanN
	Weight [3][3][7]float32
	stream cu.Stream
}

// Stencil adds the stencil result to out
func NewStencil3D(tag, unit string, in nimble.ChanN) *Stencil3D {
	r := in.NewReader() // TODO: buffer
	w := nimble.MakeChanN(3, tag, unit, r.Mesh(), in.MemType(), -1)
	return &Stencil3D{in: r, out: w, stream: cu.StreamCreate()}
}

func (s *Stencil3D) Exec() {
	//out := s.out.UnsafeData()
}

func (s *Stencil3D) Output() nimble.ChanN {
	return s.out
}

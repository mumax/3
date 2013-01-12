package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/gpu/ptx"
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
)

type Stencil struct {
	in     nimble.RChan1
	out    nimble.Chan1
	Weight [7]float32
	stream cu.Stream
}

func NewStencil(tag, unit string, in nimble.Chan, Weight [7]float32) *Stencil {
	r := in.ChanN().Chan1().NewReader() // TODO: buffer
	w := nimble.MakeChan1(tag, unit, r.Mesh, in.MemType(), -1)
	return &Stencil{r, w, Weight, cu.StreamCreate()}
}

func (s *Stencil) Exec() {
	dst := s.out.UnsafeData().Device()
	dst.Memset(0)
	src := s.in.UnsafeData().Device()
	StencilAdd(dst, src, s.out.Mesh, &s.Weight, s.stream)
}

func (s *Stencil) Output() nimble.Chan1 {
	return s.out
}

// StencilAdd adds to dst the stencil result.
func StencilAdd(dst, src safe.Float32s, mesh *nimble.Mesh, weight *[7]float32, stream cu.Stream) {
	core.Assert(dst.Len() == src.Len() && src.Len() == mesh.NCell())

	size := mesh.Size()
	N0, N1, N2 := size[0], size[1], size[2]
	wrap := mesh.PBC()
	core.Assert(wrap == [3]int{0, 0, 0})
	gridDim, blockDim := Make2DConf(N2, N1) // why?
	ptx.K_stencil3(dst.Pointer(), src.Pointer(),
		weight[0], weight[1], weight[2], weight[3], weight[4], weight[5], weight[6],
		wrap[0], wrap[1], wrap[2], N0, N1, N2, gridDim, blockDim)
}

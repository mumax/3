package gpu

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/nimble"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/cuda5/safe"
	"unsafe"
)

type Stencil struct {
	in     nimble.RChan1
	out    nimble.Chan1
	weight [7]float32
	stream cu.Stream
}

func NewStencil(tag, unit string, in nimble.Chan, weight [7]float32) *Stencil {
	r := in.ChanN().Chan1().NewReader() // TODO: buffer
	w := nimble.MakeChan1(tag, unit, r.Mesh, in.MemType(), -1)
	return &Stencil{r, w, weight, cu.StreamCreate()}
}

func (s *Stencil) Exec() {
	N := s.out.NCell()
	CalcStencil(s.out.UnsafeData().Device(), s.in.UnsafeData().Device(), s.out.Mesh, &s.weight, s.stream)
	//s.out.WriteDone()
	//s.in.ReadDone()
}

func (s *Stencil) Output() nimble.Chan1 {
	return s.out
}

func CalcStencil(dst, src safe.Float32s, mesh *nimble.Mesh, weight *[7]float32, stream cu.Stream) {
	core.Assert(dst.Len() == src.Len() && src.Len() == mesh.NCell())

	size := mesh.Size()
	N0, N1, N2 := size[0], size[1], size[2]
	wrap := mesh.PBC()
	core.Assert(wrap == [3]int{0, 0, 0})

	gridDim, blockDim := Make2DConf(N2, N1) // why?

	dptr := dst.Pointer()
	sptr := src.Pointer()
	args := []unsafe.Pointer{
		unsafe.Pointer(&dptr),
		unsafe.Pointer(&sptr),
		unsafe.Pointer(&(*weight)[0]),
		unsafe.Pointer(&(*weight)[1]),
		unsafe.Pointer(&(*weight)[2]),
		unsafe.Pointer(&(*weight)[3]),
		unsafe.Pointer(&(*weight)[4]),
		unsafe.Pointer(&(*weight)[5]),
		unsafe.Pointer(&(*weight)[6]),
		unsafe.Pointer(&wrap[0]),
		unsafe.Pointer(&wrap[1]),
		unsafe.Pointer(&wrap[2]),
		unsafe.Pointer(&N0),
		unsafe.Pointer(&N1),
		unsafe.Pointer(&N2)}

	code := PTXLoad("stencil3")
	shmem := 0
	cu.LaunchKernel(code, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
	stream.Synchronize()
}

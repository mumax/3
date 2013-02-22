package cuda

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/kernel"
	"code.google.com/p/mx3/util"
)

type Stencil struct {
	Weight [7]float32
}

func (s *Stencil) Exec(dst *data.Slice, src *data.Slice) {
	Memset(dst, 0)
	stencilAdd(dst, src, &s.Weight)
}

// adds to dst the stencil result.
func stencilAdd(dst, src *data.Slice, weight *[7]float32) {
	mesh := dst.Mesh()
	util.Argument(dst.Len() == src.Len() && src.Len() == mesh.NCell())
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Argument(dst.DevPtr(0) != src.DevPtr(0)) // no in-place operation

	size := mesh.Size()
	N0, N1, N2 := size[0], size[1], size[2]
	wrap := mesh.PBC()
	util.Assert(wrap == [3]int{0, 0, 0})
	gridDim, blockDim := Make2DConf(N2, N1)

	kernel.K_stencil3(dst.DevPtr(0), src.DevPtr(0),
		weight[0], weight[1], weight[2], weight[3], weight[4], weight[5], weight[6],
		wrap[0], wrap[1], wrap[2], N0, N1, N2, gridDim, blockDim)
}

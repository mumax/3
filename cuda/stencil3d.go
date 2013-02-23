package cuda

import "code.google.com/p/mx3/data"

type Stencil3D struct {
	Weight [3][3][7]float32
}

func (s *Stencil3D) Exec(out *data.Slice, in *data.Slice) {
	Memset(out, 0, 0, 0)
	for di := 0; di < 3; di++ {
		dst := out.Comp(di)
		for si := 0; si < 3; si++ {
			src := in.Comp(si)
			stencilAdd(dst, src, &(s.Weight[di][si]))
		}
	}
}

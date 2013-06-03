package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// special bufferedQuant to store magnetization.
// Set is overridden to stencil m with geometry.
type magnetization struct {
	bufferedQuant
}

func (q *magnetization) init() {
	q.bufferedQuant = buffered(cuda.NewSlice(3, Mesh()), "m", "")
}

// overrides normal set to allow stencil ops
func (b *magnetization) Set(src *data.Slice) {
	if src.Mesh().Size() != b.buffer.Mesh().Size() {
		src = data.Resample(src, b.buffer.Mesh().Size())
	}
	stencil(src, vol.host)
	data.Copy(b.buffer, src)
}

func (m *magnetization) stencil(s *data.Slice) {
	if s.IsNil() {
		return
	}
	h := hostBuf(m.NComp(), m.Mesh())
	data.Copy(h, m.buffer)
	stencil(h, s)
	data.Copy(m.buffer, h)
}

func stencil(dst, stencil *data.Slice) {
	if stencil.IsNil() {
		return
	}
	util.Argument(stencil.NComp() == 1)
	s := stencil.Host()[0]
	d := dst.Host()
	for c := range d {
		for i := range d[c] {
			d[c][i] *= s[i]
		}
	}
}

// TODO: normalize M after set

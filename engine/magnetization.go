package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

// special bufferedQuant to store magnetization.
// Set is overridden to stencil m with geometry.
type magnetization struct {
	bufferedQuant
}

func (q *magnetization) init() {
	q.bufferedQuant = *newBuffered(cuda.NewSlice(3, Mesh()), "m", "")
}

func (b *magnetization) Set(src *data.Slice) {
	if src.Mesh().Size() != b.buffer.Mesh().Size() {
		src = data.Resample(src, b.buffer.Mesh().Size())
	}
	// stencil here
	data.Copy(b.buffer, src)
}

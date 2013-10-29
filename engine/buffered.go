package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// A buffered quantity is stored in GPU memory at all times.
type buffered struct {
	info
	buffer *data.Slice
}

// init metadata but does not allocate yet
func (b *buffered) init(nComp int, name, unit, doc_ string, mesh *data.Mesh) {
	b.info = Info(nComp, name, unit, mesh)
}

// allocate storage (not done by init)
func (q *buffered) alloc() {
	q.buffer = cuda.NewSlice(3, q.Mesh())
}

// get buffer (on GPU, no need to recycle)
func (b *buffered) Slice() (q *data.Slice, recycle bool) {
	return b.buffer, false
}

// Set the value of one cell.
func (b *buffered) SetCell(ix, iy, iz int, v ...float64) {
	nComp := b.NComp()
	util.Argument(len(v) == nComp)
	for c := 0; c < nComp; c++ {
		cuda.SetCell(b.buffer, c, ix, iy, iz, float32(v[c]))
	}
}

// Get the value of one cell.
func (b *buffered) GetCell(comp, ix, iy, iz int) float64 {
	return float64(cuda.GetCell(b.buffer, comp, ix, iy, iz))
}

func (q *buffered) Region(r int) *inRegion { return &inRegion{q, r} }
func (m *buffered) TableData() []float64   { return Average(m) }

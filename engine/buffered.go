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
func (q *buffered) init(nComp int, name, unit string, mesh *data.Mesh) {
	q.info = Info(nComp, name, unit, mesh)
}

// allocate storage (not done by init, as mesh size may not yet be known then)
func (q *buffered) alloc() {
	q.buffer = cuda.NewSlice(3, q.Mesh().Size())
}

// get buffer (on GPU, no need to recycle)
func (q *buffered) Slice() (s *data.Slice, recycle bool) {
	return q.buffer, false
}

// Set the value of one cell.
func (q *buffered) SetCell(ix, iy, iz int, v ...float64) {
	nComp := q.NComp()
	util.Argument(len(v) == nComp)
	for c := 0; c < nComp; c++ {
		cuda.SetCell(q.buffer, c, ix, iy, iz, float32(v[c]))
	}
}

// Get the value of one cell.
func (q *buffered) GetCell(comp, ix, iy, iz int) float64 {
	return float64(cuda.GetCell(q.buffer, comp, ix, iy, iz))
}

func (q *buffered) Region(r int) *sliceInRegion { return &sliceInRegion{q, r} }

func (q *buffered) TableData() []float64 { return Average(q) }

func (q *buffered) String() string { return util.Sprint(q.buffer.HostCopy()) }

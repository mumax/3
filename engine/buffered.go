package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

// A buffered quantity is stored in GPU memory at all times.
// E.g.: magnetization.
type bufferedQuant struct {
	autosave
	buffer *data.Slice
}

// constructor
func buffered(slice *data.Slice, name, unit string) bufferedQuant {
	return bufferedQuant{newAutosave(slice.NComp(), name, unit, slice.Mesh()), slice}
}

// notify that it may need to be saved.
func (b *bufferedQuant) notifySave(cansave bool) {
	if cansave && b.needSave() {
		Save(b)
		b.saved()
	}
}

// get buffer (on GPU, no need to recycle)
func (b *bufferedQuant) Get() (q *data.Slice, recycle bool) {
	b.init()
	return b.buffer, false
}

// get buffer (on GPU, no need to recycle)
func (b *bufferedQuant) GetGPU() (q *data.Slice, recycle bool) {
	b.init()
	return b.buffer, false
}

// Replace the data by src. Auto rescales if needed.
func (b *bufferedQuant) Set(src *data.Slice) {
	b.init()
	if src.Mesh().Size() != b.buffer.Mesh().Size() {
		src = data.Resample(src, b.buffer.Mesh().Size())
	}
	data.Copy(b.buffer, src)
}

// Set the value of one cell.
func (b *bufferedQuant) SetCell(ix, iy, iz int, v ...float64) {
	b.init()
	nComp := b.NComp()
	util.Argument(len(v) == nComp)
	for c := 0; c < nComp; c++ {
		cuda.SetCell(b.buffer, util.SwapIndex(c, nComp), iz, iy, ix, float32(v[c]))
	}
}

// Get the value of one cell
func (b *bufferedQuant) GetCell(comp, ix, iy, iz int) float64 {
	b.init()
	return float64(cuda.GetCell(b.buffer, util.SwapIndex(comp, b.NComp()), iz, iy, ix))
}

// Shift the data over (shx, shy, shz cells), clamping boundary values.
// Typically used in a PostStep function to center the magnetization on
// the simulation window.
func (b *bufferedQuant) Shift(shx, shy, shz int) {
	b.init()
	m2 := cuda.GetBuffer(1, b.buffer.Mesh())
	defer cuda.RecycleBuffer(m2)
	for c := 0; c < b.NComp(); c++ {
		comp := b.buffer.Comp(c)
		cuda.Shift(m2, comp, [3]int{shz, shy, shx}) // ZYX !
		data.Copy(comp, m2)
	}
}

// Allocate buffer data (on GPU) if not yet done so.
// Used by masks, who are not allocated before needed.
func (m *bufferedQuant) init() {
	if m.buffer.DevPtr(0) == nil {
		m.buffer = cuda.NewSlice(m.NComp(), m.mesh) // could alloc only needed components...
		cuda.Memset(m.buffer, 1, 1, 1)              // default value for mask.
	}
}

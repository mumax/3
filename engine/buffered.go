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

func (b *bufferedQuant) Get() (q *data.Slice, recycle bool) {
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

func (m *bufferedQuant) init() {
	if m.isNil() {
		m.buffer = cuda.NewSlice(m.NComp(), m.mesh) // could alloc only needed components...
		cuda.Memset(m.buffer, 1, 1, 1)              // default value for mask.
	}
}

func (m *bufferedQuant) isNil() bool {
	return m.buffer.DevPtr(0) == nil
}

func (b *bufferedQuant) SetCell(ix, iy, iz int, v ...float64) {
	b.init()
	nComp := b.NComp()
	util.Argument(len(v) == nComp)
	for c := 0; c < nComp; c++ {
		cuda.SetCell(b.buffer, util.SwapIndex(c, nComp), iz, iy, ix, float32(v[c]))
	}
}

func (b *bufferedQuant) GetCell(comp, ix, iy, iz int) float64 {
	b.init()
	return float64(cuda.GetCell(b.buffer, util.SwapIndex(comp, b.NComp()), iz, iy, ix))
}

// Shift the data over (shx, shy, shz cells), clamping boundary values.
// Typically used in a PostStep function to center the magnetization on
// the simulation window.
func (b *bufferedQuant) Shift(shx, shy, shz int) {
	m2 := cuda.GetBuffer(1, b.buffer.Mesh())
	defer cuda.RecycleBuffer(m2)
	for c := 0; c < b.NComp(); c++ {
		comp := b.buffer.Comp(c)
		cuda.Shift(m2, comp, [3]int{shz, shy, shx}) // ZYX !
		data.Copy(comp, m2)
	}
}

// Get a host copy.
// TODO: assume it can be called from another thread,
// transfer asynchronously + sync
func (b *bufferedQuant) Download() *data.Slice {
	return b.buffer.HostCopy()
}

func (b *bufferedQuant) GetSlice() (s *data.Slice, recycle bool) {
	return b.buffer, false
}
func (b *bufferedQuant) Average() []float64 {
	return average(b)
}

func (b *bufferedQuant) getGPU() (s *data.Slice, mustRecycle bool) {
	return b.buffer, false
}

const (
	X = 0
	Y = 1
	Z = 2
)

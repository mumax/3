package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"reflect"
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

// get buffer (on GPU, no need to recycle)
func (b *bufferedQuant) Get() (q *data.Slice, recycle bool) {
	b.alloc()
	return b.buffer, false
}

// get buffer (on GPU, no need to recycle)
func (b *bufferedQuant) GetGPU() (q *data.Slice, recycle bool) {
	b.alloc()
	return b.buffer, false
}

// Replace the data by src. Auto rescales if needed.
func (b *bufferedQuant) Set(src *data.Slice) {
	b.alloc()
	if src.Mesh().Size() != b.buffer.Mesh().Size() {
		src = data.Resample(src, b.buffer.Mesh().Size())
	}
	data.Copy(b.buffer, src)
}

// Set the value of one cell.
func (b *bufferedQuant) SetCell(ix, iy, iz int, v ...float64) {
	b.alloc()
	nComp := b.NComp()
	util.Argument(len(v) == nComp)
	for c := 0; c < nComp; c++ {
		cuda.SetCell(b.buffer, util.SwapIndex(c, nComp), iz, iy, ix, float32(v[c]))
	}
}

// Get the value of one cell
func (b *bufferedQuant) GetCell(comp, ix, iy, iz int) float64 {
	b.alloc()
	return float64(cuda.GetCell(b.buffer, util.SwapIndex(comp, b.NComp()), iz, iy, ix))
}

func (b *bufferedQuant) Save() {
	save(b)
}

func (b *bufferedQuant) SaveAs(fname string) {
	saveAs(b, fname)
}

// Shift the data over (shx, shy, shz cells), clamping boundary values.
// Typically used in a PostStep function to center the magnetization on
// the simulation window.
func (b *bufferedQuant) Shift(shx, shy, shz int) {
	b.alloc()
	m2 := cuda.GetBuffer(1, b.buffer.Mesh())
	defer cuda.RecycleBuffer(m2)
	for c := 0; c < b.NComp(); c++ {
		comp := b.buffer.Comp(c)
		cuda.Shift(m2, comp, [3]int{shz, shy, shx}) // ZYX !
		data.Copy(comp, m2)
	}
}

func (b *bufferedQuant) GetVec() []float64 {
	return Average(b)
}

// Allocate buffer data (on GPU) if not yet done so.
// Used by masks, who are not allocated before needed.
func (m *bufferedQuant) alloc() {
	if m.buffer.IsNil() {
		m.buffer = cuda.NewSlice(m.NComp(), m.mesh) // could alloc only needed components...
	}
}

func (b *bufferedQuant) SetValue(v interface{})  { b.Set(v.(*data.Slice)) }
func (b *bufferedQuant) Eval() interface{}       { return b }
func (b *bufferedQuant) InputType() reflect.Type { return reflect.TypeOf(new(data.Slice)) }

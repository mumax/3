package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"path"
)

// Output Handle for a quantity that is stored on the GPU.
type buffered struct {
	buffer *data.Slice
	autosave
}

func newBuffered(slice *data.Slice, name string) *buffered {
	return &buffered{slice, autosave{name: name}}
}

// notify the handle that it may need to be saved
func (b *buffered) touch(goodstep bool) {
	if goodstep && b.needSave() {
		b.Save()
		b.saved()
	}
}

func (b *buffered) NComp() int { return b.buffer.NComp() }

// Save once, with automatically assigned file name.
func (b *buffered) Save() {
	goSaveCopy(b.fname(), b.buffer, Time)
	b.autonum++
}

// Save once, with given file name.
func (b *buffered) SaveAs(fname string) {
	if !path.IsAbs(fname) {
		fname = OD + fname
	}
	goSaveCopy(fname, b.buffer, Time)
}

// Get a host copy.
// TODO: assume it can be called from another thread,
// transfer asynchronously.
func (m *buffered) Download() *data.Slice {
	return m.buffer.HostCopy()
}

// Returns the average over all cells.
// TODO: does not belong here
func (b *buffered) Average() []float64 {
	return average(b.buffer)
}

// Returns the maximum norm of a vector field.
// TODO: only for vectors
// TODO: does not belong here
func (b *buffered) MaxNorm() float64 {
	return cuda.MaxVecNorm(b.buffer)
}

// average in userspace XYZ order
// does not yet take into account volume.
// pass volume parameter, possibly nil?
func average(b *data.Slice) []float64 {
	nComp := b.NComp()
	avg := make([]float64, nComp)
	for i := range avg {
		I := swapIndex(i, nComp)
		avg[i] = float64(cuda.Sum(b.Comp(I))) / float64(b.Mesh().NCell())
	}
	return avg
}

// Replace the data by src. Auto rescales if needed.
func (m *buffered) Set(src *data.Slice) {
	if src.Mesh().Size() != m.buffer.Mesh().Size() {
		src = data.Resample(src, m.buffer.Mesh().Size())
	}
	data.Copy(m.buffer, src)
}

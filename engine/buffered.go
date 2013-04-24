package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"path"
)

// function that sets ("updates") quantity stored in dst
type updFunc func(dst *data.Slice)

// Output Handle for a quantity that is stored on the GPU.
type buffered struct {
	*data.Synced
	updFn updFunc
	autosave
}

func newBuffered(synced *data.Synced, name string, f updFunc) *buffered {
	b := new(buffered)
	b.Synced = synced
	b.name = name
	b.updFn = f
	return b
}

func (b *buffered) update(goodstep bool) {
	dst := b.Write()
	b.updFn(dst)
	b.WriteDone()
	b.touch(goodstep)
}

// notify the handle that it may need to be saved
func (b *buffered) touch(goodstep bool) {
	if goodstep && b.needSave() {
		b.Save()
		b.saved()
	}
}

// Save once, with automatically assigned file name.
func (b *buffered) Save() {
	goSaveCopy(b.fname(), b.Read(), Time)
	b.autonum++
}

// Save once, with given file name.
func (b *buffered) SaveAs(fname string) {
	if !path.IsAbs(fname) {
		fname = OD + fname
	}
	goSaveCopy(fname, b.Read(), Time)
}

// Get a host copy.
// TODO: assume it can be called from another thread,
// transfer asynchronously.
func (m *buffered) Download() *data.Slice {
	m_ := m.Read()
	host := m_.HostCopy()
	m.ReadDone()
	return host
}

// Replace the data by src. Auto rescales if needed.
func (m *buffered) Upload(src *data.Slice) {
	if src.Mesh().Size() != m.Mesh().Size() {
		src = data.Resample(src, m.Mesh().Size())
	}
	m_ := m.Write()
	data.Copy(m_, src)
	m.WriteDone()
}

// Memset with synchronization.
func (b *buffered) memset(val ...float32) {
	s := b.Write()
	cuda.Memset(s, val...)
	b.WriteDone()
}

// Normalize with synchronization.
func (b *buffered) normalize() {
	s := b.Write()
	cuda.Normalize(s)
	b.WriteDone()
}

// Returns the average over all cells.
func (b *buffered) Average() []float64 {
	return average(b.Synced)
}

// Returns the maximum norm of a vector field.
func (b *buffered) MaxNorm() float64 {
	s := b.Read()
	defer b.ReadDone()
	return cuda.MaxVecNorm(s)
}

// average in userspace XYZ order
// does not yet take into account volume.
// pass volume parameter, possibly nil?
func average(b *data.Synced) []float64 {
	s := b.Read()
	defer b.ReadDone()
	nComp := s.NComp()
	avg := make([]float64, nComp)
	for i := range avg {
		I := swapIndex(i, nComp)
		avg[i] = float64(cuda.Sum(s.Comp(I))) / float64(s.Mesh().NCell())
	}
	return avg
}

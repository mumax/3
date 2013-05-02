package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"path"
)

// Output Handle for a quantity that is stored on the GPU.
type buffered struct {
	autosave
	buffer *data.Slice
	shiftc [3]int  // shift count (total shifted cells in each direction)
	shift  *scalar // returns total shift in meters.
}

func newBuffered(slice *data.Slice, name, unit string) *buffered {
	b := &buffered{newAutosave(name, unit, slice.Mesh()), slice, [3]int{}, nil}
	b.shift = newScalar(3, name+"_shift", "m", b.getShift)
	return b
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

func (b *buffered) SetFile(fname string) {
	util.FatalErr(b.setFile(fname))
}

func (b *buffered) setFile(fname string) error {
	m, _, err := data.ReadFile(fname)
	if err != nil {
		return err
	}
	b.Set(m)
	return nil
}

//
func (b *buffered) SetCell(ix, iy, iz int, v ...float64) {
	nComp := b.NComp()
	util.Argument(len(v) == nComp)
	for c := 0; c < nComp; c++ {
		cuda.SetCell(b.buffer, swapIndex(c, nComp), iz, iy, ix, float32(v[c]))
	}
}

// returns shift in m
func (s *buffered) getShift() []float64 {
	c := s.mesh.CellSize()
	return []float64{c[2] * float64(s.shiftc[0]), c[1] * float64(s.shiftc[1]), c[0] * float64(s.shiftc[2])}
}

// Shift the data over (shx, shy, shz cells), clamping boundary values.
// Typically used in a PostStep function to center the magnetization on
// the simulation window.
func (b *buffered) Shift(shx, shy, shz int) {
	m := b.buffer
	m2 := cuda.GetBuffer(1, m.Mesh())
	defer cuda.RecycleBuffer(m2)
	b.shiftc[X] += shx
	b.shiftc[Y] += shy
	b.shiftc[Z] += shz
	for c := 0; c < m.NComp(); c++ {
		cuda.Shift(m2, m.Comp(c), [3]int{shz, shy, shx}) // ZYX !
		data.Copy(m.Comp(c), m2)
	}
}

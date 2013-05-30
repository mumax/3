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
	shiftc [3]int // shift count (total shifted cells in each direction)
	//shift  *scalar // returns total shift in meters.
}

func buffered(slice *data.Slice, name, unit string) bufferedQuant {
	return bufferedQuant{newAutosave(slice.NComp(), name, unit, slice.Mesh()), slice, [3]int{}}
	//b.shift = newScalar(3, name+"_shift", "m", b.getShift)
	//return b
}

// notify that it may need to be saved.
func (b *bufferedQuant) notifySave(goodstep bool) {
	if goodstep && b.needSave() {
		b.Save()
		b.saved()
	}
}

// Replace the data by src. Auto rescales if needed.
func (b *bufferedQuant) Set(src *data.Slice) {
	if src.Mesh().Size() != b.buffer.Mesh().Size() {
		src = data.Resample(src, b.buffer.Mesh().Size())
	}
	data.Copy(b.buffer, src)
}

// TODO: read(file)
//func (b *bufferedQuant) SetFile(fname string) {
//	util.FatalErr(b.setFile(fname))
//}

//func (b *bufferedQuant) setFile(fname string) error {
//	m, _, err := data.ReadFile(fname)
//	if err != nil {
//		return err
//	}
//	b.Set(m)
//	return nil
//}

//
func (b *bufferedQuant) SetCell(ix, iy, iz int, v ...float64) {
	nComp := b.NComp()
	util.Argument(len(v) == nComp)
	for c := 0; c < nComp; c++ {
		cuda.SetCell(b.buffer, util.SwapIndex(c, nComp), iz, iy, ix, float32(v[c]))
	}
}

func (b *bufferedQuant) GetCell(comp, ix, iy, iz int) float64 {
	return float64(cuda.GetCell(b.buffer, util.SwapIndex(comp, b.NComp()), iz, iy, ix))
}

// Shift the data over (shx, shy, shz cells), clamping boundary values.
// Typically used in a PostStep function to center the magnetization on
// the simulation window.
func (b *bufferedQuant) Shift(shx, shy, shz int) {
	m2 := cuda.GetBuffer(1, b.buffer.Mesh())
	defer cuda.RecycleBuffer(m2)
	b.shiftc[X] += shx
	b.shiftc[Y] += shy
	b.shiftc[Z] += shz
	for c := 0; c < b.NComp(); c++ {
		comp := b.buffer.Comp(c)
		cuda.Shift(m2, comp, [3]int{shz, shy, shx}) // ZYX !
		data.Copy(comp, m2)
	}
}

// total shift in meters
//func (b *bufferedQuant) ShiftDistance() *scalar {
//	return b.shift
//}

// returns shift of simulation window in m
func (b *bufferedQuant) getShift() []float64 {
	c := b.mesh.CellSize()
	return []float64{-c[2] * float64(b.shiftc[0]), -c[1] * float64(b.shiftc[1]), -c[0] * float64(b.shiftc[2])}
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

func (b *bufferedQuant) Save() {
	saveAs(b, b.autoFname())
}

func (b *bufferedQuant) getGPU() (s *data.Slice, mustRecycle bool) {
	return b.buffer, false
}

const (
	X = 0
	Y = 1
	Z = 2
)

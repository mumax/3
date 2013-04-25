package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

type Magnetization struct {
	*buffered //  TODO: buffered <-> settable?
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
		cuda.SetCell(b.Slice, swapIndex(c, nComp), iz, iy, ix, float32(v[c]))
	}
}

// Shift the data over (shx, shy, shz cells), clamping boundary values.
// Typically used in a PostStep function to center the magnetization on
// the simulation window.
func (b *buffered) Shift(shx, shy, shz int) {
	m := b.Slice
	m2 := cuda.GetBuffer(1, m.Mesh())
	defer cuda.RecycleBuffer(m2)

	for c := 0; c < m.NComp(); c++ {
		cuda.Shift(m2, m.Comp(c), [3]int{shz, shy, shx})
		data.Copy(m.Comp(c), m2)
	}
}

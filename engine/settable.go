package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

type Settable struct {
	*buffered //  TODO: buffered <-> settable?
}

func (b *Settable) SetFile(fname string) {
	util.FatalErr(b.setFile(fname))
}

func (b *Settable) setFile(fname string) error {
	m, _, err := data.ReadFile(fname)
	if err != nil {
		return err
	}
	b.Set(m)
	return nil
}

//
func (b *Settable) SetCell(ix, iy, iz int, v ...float64) {
	nComp := b.NComp()
	util.Argument(len(v) == nComp)
	for c := 0; c < nComp; c++ {
		cuda.SetCell(b.buffer, swapIndex(c, nComp), iz, iy, ix, float32(v[c]))
	}
}

var (
	Shift = newScalar(3, "shift", "m", func() []float64 {
		c := CellSize()
		return []float64{float64(shift[X]) * c[X], float64(shift[Y]) * c[Y], float64(shift[Z]) * c[Z]}
	})
	shift [3]int
)

// Shift the data over (shx, shy, shz cells), clamping boundary values.
// Typically used in a PostStep function to center the magnetization on
// the simulation window.
func (b *Settable) Shift(shx, shy, shz int) {
	m := b.buffer
	m2 := cuda.GetBuffer(1, m.Mesh())
	defer cuda.RecycleBuffer(m2)
	shift[X] += shx
	shift[Y] += shy
	shift[Z] += shz
	for c := 0; c < m.NComp(); c++ {
		cuda.Shift(m2, m.Comp(c), [3]int{shz, shy, shx}) // ZYX !
		data.Copy(m.Comp(c), m2)
	}
}

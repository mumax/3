package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	TotalShift               float64        // accumulated window shift (X) in meter
	ShiftClampL, ShiftClampR float32        // when shifting m, put these value at the left/right edge.
	shiftM                   bool    = true // should shift act on magnetization?
)

func GetShiftPos() float64 { return TotalShift }

// shift the simulation window over dx cells in X direction
func Shift(dx int) {

	// shift m
	if shiftM {
		m := M.Buffer()
		m2 := cuda.Buffer(1, m.Mesh())
		defer cuda.Recycle(m2)
		for c := 0; c < m.NComp(); c++ {
			comp := m.Comp(c)
			cuda.ShiftX(m2, comp, dx, ShiftClampL, ShiftClampR)
			data.Copy(comp, m2) // str0 ?
		}
	}

	//	regions.shift(dx, 0, 0)
	//	geometry.shift(dx)

	TotalShift += float64(dx) * Mesh().CellSize()[X]
}

func (b *Regions) shift(shx, shy, shz int) {
	r1 := b.Gpu()
	r2 := cuda.NewBytes(b.Mesh()) // TODO: somehow recycle
	defer r2.Free()
	cuda.ShiftBytes(r2, r1, b.Mesh(), [3]int{shx, shy, shz})
	r1.Copy(r2)
}

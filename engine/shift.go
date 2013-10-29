package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	totalShift float64 // accumulated window shift (X) in meter
)

func GetShiftPos() float64 { return totalShift }

func updateShift(dir, sign int) {
	totalShift -= float64(sign) * Mesh().CellSize()[dir] // window left means wall right: minus sign
}

// shift the simulation window over dx cells in user X direction
func Shift(dx int) {
	shift(M.buffer, 0, 0, dx)
	regions.shift(0, 0, dx)
	geometry.shift(dx)
	updateShift(X, -dx)
}

// Shift the data over (shx, shy, shz cells), clamping boundary values.
// Typically used in a PostStep function to center the magnetization on
// the simulation window.
func shift(s *data.Slice, shx, shy, shz int) {
	m2 := cuda.Buffer(1, s.Mesh())
	defer cuda.Recycle(m2)
	for c := 0; c < s.NComp(); c++ {
		comp := s.Comp(c)
		cuda.Shift(m2, comp, [3]int{shx, shy, shz})
		data.Copy(comp, m2)
	}
}

func (b *Regions) shift(shx, shy, shz int) {
	r1 := b.Gpu()
	r2 := cuda.NewBytes(b.Mesh()) // TODO: somehow recycle
	defer r2.Free()
	cuda.ShiftBytes(r2, r1, b.Mesh(), [3]int{shx, shy, shz})
	r1.Copy(r2)
}

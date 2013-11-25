package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	TotalShift                      float64                        // accumulated window shift (X) in meter
	ShiftMagL, ShiftMagR            data.Vector                    // when shifting m, put these value at the left/right edge.
	ShiftM, ShiftGeom, ShiftRegions bool        = true, true, true // should shift act on magnetization, geometry, regions?
)

func GetShiftPos() float64 { return TotalShift }

// shift the simulation window over dx cells in X direction
func Shift(dx int) {

	util.Argument(dx == 1 || dx == -1)

	// shift m
	if ShiftM {
		shiftMag(M.Buffer(), dx)
	}

	//	regions.shift(dx, 0, 0)
	if ShiftGeom {
		geometry.shift(dx)
	}

	TotalShift += float64(dx) * Mesh().CellSize()[X]
}

func shiftMag(m *data.Slice, dx int) {
	m2 := cuda.Buffer(1, m.Mesh())
	defer cuda.Recycle(m2)
	for c := 0; c < m.NComp(); c++ {
		comp := m.Comp(c)
		cuda.ShiftX(m2, comp, dx, float32(ShiftMagL[c]), float32(ShiftMagR[c]))
		data.Copy(comp, m2) // str0 ?
	}
}

func (b *Regions) shift(shx, shy, shz int) {
	r1 := b.Gpu()
	r2 := cuda.NewBytes(b.Mesh()) // TODO: somehow recycle
	defer r2.Free()
	cuda.ShiftBytes(r2, r1, b.Mesh(), [3]int{shx, shy, shz})
	r1.Copy(r2)
}

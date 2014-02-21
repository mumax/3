package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	TotalShift                      float64                        // accumulated window shift (X) in meter
	ShiftMagL, ShiftMagR            data.Vector                    // when shifting m, put these value at the left/right edge.
	ShiftM, ShiftGeom, ShiftRegions bool        = true, true, true // should shift act on magnetization, geometry, regions?
)

func init() {
	DeclFunc("Shift", Shift, "Shifts the simulation by +1/-1 cells along X")
	DeclVar("ShiftMagL", &ShiftMagL, "Upon shift, insert this magnetization from the left")
	DeclVar("ShiftMagR", &ShiftMagR, "Upon shift, insert this magnetization from the right")
	DeclVar("ShiftM", &ShiftM, "Whether Shift() acts on magnetization")
	DeclVar("ShiftGeom", &ShiftGeom, "Whether Shift() acts on geometry")
	DeclVar("ShiftRegions", &ShiftRegions, "Whether Shift() acts on regions")
	DeclVar("TotalShift", &TotalShift, "Amount by which the simulation has been shifted (m).")
}

// position of the window lab frame
func GetShiftPos() float64 { return -TotalShift }

// shift the simulation window over dx cells in X direction
func Shift(dx int) {
	//util.Argument(dx == 1 || dx == -1) // one cell at a time please

	TotalShift += float64(dx) * Mesh().CellSize()[X] // needed to re-init geom, regions
	if ShiftM {
		shiftMag(M.Buffer(), dx) // TODO: M.shift?
	}
	if ShiftRegions {
		regions.shift(dx)
	}
	if ShiftGeom {
		geometry.shift(dx)
	}
	M.normalize()
}

func shiftMag(m *data.Slice, dx int) {
	m2 := cuda.Buffer(1, m.Size())
	defer cuda.Recycle(m2)
	for c := 0; c < m.NComp(); c++ {
		comp := m.Comp(c)
		cuda.ShiftX(m2, comp, dx, float32(ShiftMagL[c]), float32(ShiftMagR[c]))
		data.Copy(comp, m2) // str0 ?
	}
}

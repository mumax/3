package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	TotalShiftY                      float64                        // accumulated window shift (X) in meter
	ShiftMagT, ShiftMagB            data.Vector                    // when shifting m, put these value at the top/bottom edge.
	ShiftMY, ShiftGeomY, ShiftRegionsY bool        = true, true, true // should shift act on magnetization, geometry, regions?
)

func init() {
	DeclFunc("ShiftY", ShiftY, "Shifts the simulation by +1/-1 cells along Y")
	DeclVar("ShiftMagT", &ShiftMagT, "Upon shift, insert this magnetization from the top")
	DeclVar("ShiftMagB", &ShiftMagB, "Upon shift, insert this magnetization from the bottom")
	DeclVar("ShiftMY", &ShiftMY, "Whether Shift() acts on magnetization")
	DeclVar("ShiftGeomY", &ShiftGeomY, "Whether Shift() acts on geometry")
	DeclVar("ShiftRegionsY", &ShiftRegionsY, "Whether Shift() acts on regions")
	DeclVar("TotalShiftY", &TotalShiftY, "Amount by which the simulation has been Y shifted (m).")
}

// position of the window lab frame
func GetShiftPosY() float64 { return -TotalShiftY }

// shift the simulation window over dy cells in Y direction
func ShiftY(dy int) {
	TotalShiftY += float64(dy) * Mesh().CellSize()[Y] // needed to re-init geom, regions
	if ShiftMY {
		shiftMagY(M.Buffer(), dy) // TODO: M.shift?
	}
	if ShiftRegionsY {
		regions.shifty(dy)
	}
	if ShiftGeomY {
		geometry.shiftY(dy)
	}
	M.normalize()
}

func shiftMagY(m *data.Slice, dy int) {
	m2 := cuda.Buffer(1, m.Size())
	defer cuda.Recycle(m2)
	for c := 0; c < m.NComp(); c++ {
		comp := m.Comp(c)
		cuda.ShiftY(m2, comp, dy, float32(ShiftMagB[c]), float32(ShiftMagT[c]))
		data.Copy(comp, m2) // str0 ?
	}
}

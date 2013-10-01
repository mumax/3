package engine

var (
	totalShift float64 // accumulated window shift (X) in meter
	DWPos      = NewGetScalar("ext_dwpos", "m", "Position of the simulation window while following a domain wall", GetShiftPos)
)

func GetShiftPos() float64 { return totalShift }

func updateShift(dir, sign int) {
	totalShift -= float64(sign) * Mesh().CellSize()[dir] // window left means wall right: minus sign
}

func Shift(dx int) {
	M.Shift(dx, 0, 0)
	updateShift(2, -dx)
}

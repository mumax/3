package engine

var (
	totalShift float64 // accumulated window shift (X) in meter
)

func GetShiftPos() float64 { return totalShift }

func updateShift(dir, sign int) {
	totalShift -= float64(sign) * Mesh().CellSize()[dir] // window left means wall right: minus sign
}

func Shift(dx int) {
	M.Shift(dx, 0, 0)
	updateShift(2, -dx)
}

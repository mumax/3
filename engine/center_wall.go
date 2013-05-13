package engine

// Shift the magnetization to the left or right in order to keep mz close zero.
// Thus moving an up-down domain wall to the center of the simulation box.
func CenterPMAWall() {
	mz := M.Average()[Z] // TODO: optimize
	if mz > 0.01 {
		M.Shift(-1, 0, 0) // 1 cell to the left // TODO: shift others
		return
	}
	if mz < -0.01 {
		M.Shift(1, 0, 0) // 1 cell to the right
	}
}

func init() {
	parser.AddFunc("centerPMAwall", CenterPMAWall)
}

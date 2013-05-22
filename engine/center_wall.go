package engine

import "fmt"

// This post-step function centers the simulation window on a domain wall between up-down (or down-up) domains (like in perpendicular media). E.g.:
// 	PostStep(CenterPMAWall)
func CenterPMAWall() {
	mz := M.Average()[Z]           // TODO: optimize
	tolerance := 4 / float64(Nx()) // 2 * expected <m> change for 1 cell shift

	if mz < tolerance {
		sign := wall_left_magnetization(M.GetCell(Z, 0, Ny()/2, Nz()/2))
		M.Shift(sign, 0, 0)
		return
	}
	if mz > tolerance {
		sign := wall_left_magnetization(M.GetCell(Z, 0, Ny()/2, Nz()/2))
		M.Shift(-sign, 0, 0)
	}
}

// This post-step function centers the simulation window on a domain wall between left-right (or right-left) domains (like in soft thin films). E.g.:
// 	PostStep(CenterInplaneWall)
func CenterInplaneWall() {
	mz := M.Average()[X]           // TODO: optimize
	tolerance := 4 / float64(Nx()) // 2 * expected <m> change for 1 cell shift

	if mz < tolerance {
		sign := wall_left_magnetization(M.GetCell(X, 0, Ny()/2, Nz()/2))
		M.Shift(sign, 0, 0)
		return
	}
	if mz > tolerance {
		sign := wall_left_magnetization(M.GetCell(X, 0, Ny()/2, Nz()/2))
		M.Shift(-sign, 0, 0)
	}
}

func init() {
	world.Func("centerPMAWall", CenterPMAWall)
	world.Func("centerInplaneWall", CenterInplaneWall)
}

func wall_left_magnetization(x float64) int {
	if x > 0.6 {
		return 1
	}
	if x < -0.6 {
		return -1
	}
	panic(fmt.Errorf("center wall: unclear in which direction to shift: magnetization at border=%v", x))
}

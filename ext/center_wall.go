package ext

import (
	. "code.google.com/p/mx3/engine"
	"fmt"
)

func init() {
	World.Func("ext_centerPMAWall", CenterPMAWall)
	World.Func("ext_centerInplaneWall", CenterInplaneWall)
	World.ROnly("ext_dwspeed", &DWSpeed)
	World.ROnly("ext_dwpos", &DWPos)
}

// This post-step function centers the simulation window on a domain wall
// between up-down (or down-up) domains (like in perpendicular media). E.g.:
// 	PostStep(CenterPMAWall)
func CenterPMAWall() {
	mz := Average(&M)[2]           // TODO: optimize
	tolerance := 4 / float64(nx()) // 2 * expected <m> change for 1 cell shift

	if mz < tolerance {
		sign := wall_left_magnetization(M.GetCell(2, 0, ny()/2, nz()/2))
		M.Shift(sign, 0, 0)
		updateShift(2, sign)
		return
	}
	if mz > tolerance {
		sign := wall_left_magnetization(M.GetCell(2, 0, ny()/2, nz()/2))
		M.Shift(-sign, 0, 0)
		updateShift(2, -sign)
	}
}

// This post-step function centers the simulation window on a domain wall
// between left-right (or right-left) domains (like in soft thin films). E.g.:
// 	PostStep(CenterInplaneWall)
func CenterInplaneWall() {
	mz := Average(&M)[0]           // TODO: optimize
	tolerance := 4 / float64(nx()) // 2 * expected <m> change for 1 cell shift

	if mz < tolerance {
		sign := wall_left_magnetization(M.GetCell(0, 0, ny()/2, nz()/2))
		M.Shift(sign, 0, 0)
		updateShift(2, sign)
		return
	}
	if mz > tolerance {
		sign := wall_left_magnetization(M.GetCell(0, 0, ny()/2, nz()/2))
		M.Shift(-sign, 0, 0)
		updateShift(2, -sign)
	}
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

var (
	totalShift float64
	DWPos      = NewGetScalar("dwpos", "m", getShiftPos)
)

func updateShift(dir, sign int) {
	totalShift -= float64(sign) * Mesh().CellSize()[dir] // window left means wall right: minus sign
}

func getShiftPos() float64 {
	return totalShift
}

// used for speed
var (
	lastShift float64 // shift the last time we queried speed
	lastT     float64 // time the last time we queried speed
	lastV     float64 // speed the last time we queried speed
	DWSpeed   = NewGetScalar("dwspeed", "m/s", getShiftSpeed)
)

func getShiftSpeed() float64 {
	if lastShift != totalShift {
		lastV = (totalShift - lastShift) / (Time - lastT)
		lastShift = totalShift
		lastT = Time
	}
	return lastV
}

func nx() int { return Mesh().Size()[2] }
func ny() int { return Mesh().Size()[1] }
func nz() int { return Mesh().Size()[0] }

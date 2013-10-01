package ext

import (
	"fmt"
	"github.com/mumax/3/engine"
)

func init() {
	engine.DeclFunc("ext_centerPMAWall", CenterPMAWall, "This post-step function tries to center the simulation window on the domain wall in a perpendicular medium")
	engine.DeclFunc("ext_centerInplaneWall", CenterInplaneWall, "This post-step function tries to center the simulation window on the domain wall of an in-plane medium")
}

// This post-step function centers the simulation window on a domain wall
// between up-down (or down-up) domains (like in perpendicular media). E.g.:
// 	PostStep(CenterPMAWall)
func CenterPMAWall() {
	M := &engine.M
	mz := engine.Average(M)[2]     // TODO: optimize
	tolerance := 4 / float64(nx()) // 2 * expected <m> change for 1 cell shift

	if mz < tolerance {
		sign := wall_left_magnetization(M.GetCell(2, 0, ny()/2, nz()/2))
		engine.Shift(sign)
		return
	}
	if mz > tolerance {
		sign := wall_left_magnetization(M.GetCell(2, 0, ny()/2, nz()/2))
		engine.Shift(-sign)
	}
}

// This post-step function centers the simulation window on a domain wall
// between left-right (or right-left) domains (like in soft thin films). E.g.:
// 	PostStep(CenterInplaneWall)
func CenterInplaneWall() {
	M := &engine.M
	mz := engine.Average(M)[0]     // TODO: optimize
	tolerance := 4 / float64(nx()) // 2 * expected <m> change for 1 cell shift

	if mz < tolerance {
		sign := wall_left_magnetization(M.GetCell(0, 0, ny()/2, nz()/2))
		engine.Shift(sign)
		return
	}
	if mz > tolerance {
		sign := wall_left_magnetization(M.GetCell(0, 0, ny()/2, nz()/2))
		engine.Shift(-sign)
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

// used for speed
var (
	lastShift float64 // shift the last time we queried speed
	lastT     float64 // time the last time we queried speed
	lastV     float64 // speed the last time we queried speed
	DWSpeed   = engine.NewGetScalar("ext_dwspeed", "m/s", "Speed of the simulation window while following a domain wall", getShiftSpeed)
)

func getShiftSpeed() float64 {
	if lastShift != engine.GetShiftPos() {
		lastV = (engine.GetShiftPos() - lastShift) / (engine.Time - lastT)
		lastShift = engine.GetShiftPos()
		lastT = engine.Time
	}
	return lastV
}

func nx() int { return engine.Mesh().Size()[2] }
func ny() int { return engine.Mesh().Size()[1] }
func nz() int { return engine.Mesh().Size()[0] }

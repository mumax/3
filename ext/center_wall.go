package ext

import (
	"fmt"
	"github.com/mumax/3/data"
	"github.com/mumax/3/engine"
)

var (
	DWPos = engine.NewGetScalar("ext_dwpos", "m", "Position of the simulation window while following a domain wall", engine.GetShiftPos) // TODO: make more accurate
)

func init() {
	engine.DeclFunc("ext_centerPMAWall", CenterPMAWall, "This post-step function tries to center the simulation window on the domain wall in a perpendicular medium")
	engine.DeclFunc("ext_centerInplaneWall", CenterInplaneWall, "This post-step function tries to center the simulation window on the domain wall of an in-plane medium")
}

func centerWall(c int) {
	M := &engine.M
	mc := engine.Average(M)[c]     // TODO: optimize
	tolerance := 4 / float64(nx()) // 2 * expected <m> change for 1 cell shift

	zero := data.Vector{0, 0, 0}
	if engine.ShiftClampL == zero || engine.ShiftClampR == zero {
		sign := magsign(M.GetCell(c, 0, ny()/2, nz()/2))
		engine.ShiftClampL[c] = float64(sign)
		engine.ShiftClampR[c] = -float64(sign)
	}

	sign := magsign(engine.ShiftClampL[c])

	if mc < tolerance {
		engine.Shift(sign)
	} else if mc > tolerance {
		engine.Shift(-sign)
	}
}

// This post-step function centers the simulation window on a domain wall
// between up-down (or down-up) domains (like in perpendicular media). E.g.:
// 	PostStep(CenterPMAWall)
func CenterPMAWall() {
	centerWall(Z)
}

// This post-step function centers the simulation window on a domain wall
// between left-right (or right-left) domains (like in soft thin films). E.g.:
// 	PostStep(CenterInplaneWall)
func CenterInplaneWall() {
	centerWall(X)
}

func magsign(x float64) int {
	if x > 0.1 {
		return 1
	}
	if x < -0.1 {
		return -1
	}
	panic(fmt.Errorf("center wall: unclear in which direction to shift: magnetization at border=%v. Set ShiftMagL, ShiftMagR", x))
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

func nx() int { return engine.Mesh().Size()[X] }
func ny() int { return engine.Mesh().Size()[Y] }
func nz() int { return engine.Mesh().Size()[Z] }

const (
	X = 0
	Y = 1
	Z = 2
)

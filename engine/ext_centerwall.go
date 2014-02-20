package engine

import (
	"fmt"
	"github.com/mumax/3/data"
)

var (
	DWPos = NewGetScalar("ext_dwpos", "m", "Position of the simulation window while following a domain wall", GetShiftPos) // TODO: make more accurate
)

func init() {
	DeclFunc("ext_centerWall", CenterWall, "centerWall(c) shifts m after each step to keep m_c close to zero")
}

func centerWall(c int) {
	M := &M
	mc := sAverageUniverse(M.Buffer().Comp(c))[0]
	n := Mesh().Size()
	tolerance := 4 / float64(n[X]) // x*2 * expected <m> change for 1 cell shift

	zero := data.Vector{0, 0, 0}
	if ShiftMagL == zero || ShiftMagR == zero {
		sign := magsign(M.GetCell(c, 0, n[Y]/2, n[Z]/2))
		ShiftMagL[c] = float64(sign)
		ShiftMagR[c] = -float64(sign)
	}

	sign := magsign(ShiftMagL[c])

	//log.Println("mc", mc, "tol", tolerance)

	if mc < -tolerance {
		Shift(sign)
	} else if mc > tolerance {
		Shift(-sign)
	}
}

// This post-step function centers the simulation window on a domain wall
// between up-down (or down-up) domains (like in perpendicular media). E.g.:
// 	PostStep(CenterPMAWall)
func CenterWall(magComp int) {
	PostStep(func() { centerWall(magComp) })
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
	lastT   float64 // time the last time we queried speed
	lastP   float64 // position the last time we queried speed
	DWSpeed = NewGetScalar("ext_dwspeed", "m/s", "Speed of the simulation window while following a domain wall", getShiftSpeed)
)

func getShiftSpeed(c int) float64 {
	if lastP == 0 {
		lastP = wallPos(c)
	}
	velocity := (wallPos(c) - lastP) / (Time - lastT)
	lastP = wallPos(c)
	lastT = Time
	return velocity
}

//wall x-position
func wallPos(c int) float64 {
	m, _ := M.Slice()
	avg_m := sAverageMagnet(m)

	s := m.Size()
	Nc := s[c]
	pos := -(1. - avg_m[c]) * float64(Nc) / 2.

	cs := Mesh().CellSize()
	pos *= cs[c]

	pos += GetShiftPos() // add simulation window shift
	return pos
}

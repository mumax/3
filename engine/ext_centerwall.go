package engine

import (
	"fmt"

	"github.com/mumax/3/data"
)

var (
	DWPos   = NewScalarValue("ext_dwpos", "m", "Position of the simulation window while following a domain wall", GetShiftPos) // TODO: make more accurate
	DWxPos  = NewScalarValue("ext_dwxpos", "m", "Position of the simulation window while following a domain wall", GetDWxPos)
	DWSpeed = NewScalarValue("ext_dwspeed", "m/s", "Speed of the simulation window while following a domain wall", getShiftSpeed)
)

func init() {
	DeclFunc("ext_centerWall", CenterWall, "centerWall(c) shifts m after each step to keep m_c close to zero")
	DeclFunc("ext_centerWallInRegion", CenterWallInRegion, "centerWallInRegion(R, c) shifts m after each step to keep m_c in region R close to zero")
	DeclFunc("ext_centerWallInLayer", CenterWallInLayer, "centerWallInLayer(L, c) shifts m after each step to keep m_c in layer L close to zero")
}

func centerWall(c int) {
	M := &M
	mc := sAverageUniverse(M.Buffer().Comp(c))[0]
	n := Mesh().Size()
	tolerance := 4 / float64(n[X]) // x*2 * expected <m> change for 1 cell shift

	zero := data.Vector{0, 0, 0}
	if ShiftMagL == zero || ShiftMagR == zero {
		sign := magsign(M.GetCell(0, n[Y]/2, n[Z]/2)[c])
		ShiftMagL[c] = float64(sign)
		ShiftMagR[c] = -float64(sign)
	}

	sign := magsign(ShiftMagL[c])

	if mc < -tolerance {
		Shift(sign)
	} else if mc > tolerance {
		Shift(-sign)
	}
}

// This post-step function centers the simulation window on a domain wall
// between up-down (or down-up) domains (like in perpendicular media). E.g.:
//
//	PostStep(CenterPMAWall)
func CenterWall(magComp int) {
	PostStep(func() { centerWall(magComp) })
}

// The same functions as above, now for just one layer
func centerWallInLayerProc(layer, c int) {
	M := &M
	mc := CropLayer(M, layer).average()[c]
	n := Mesh().Size()
	tolerance := 4 / float64(n[X]) // x*2 * expected <m> change for 1 cell shift

	zero := data.Vector{0, 0, 0}
	if ShiftMagL == zero || ShiftMagR == zero {
		sign := magsign(M.GetCell(0, n[Y]/2, layer)[c])
		ShiftMagL[c] = float64(sign)
		ShiftMagR[c] = -float64(sign)
	}

	sign := magsign(ShiftMagL[c])

	if mc < -tolerance {
		Shift(sign)
	} else if mc > tolerance {
		Shift(-sign)
	}
}

func CenterWallInLayer(layer, magComp int) {
	PostStep(func() { centerWallInLayerProc(layer, magComp) })
}

// The same functions as above, now for just one region
func centerWallInRegionProc(region, c int) {
	M := &M
	mc := M.Region(region).Average()[c]
	n := Mesh().Size()
	tolerance := 4 / float64(n[X]) // x*2 * expected <m> change for 1 cell shift

	zero := data.Vector{0, 0, 0}
	if ShiftMagL == zero || ShiftMagR == zero {
		sign := magsign(M.GetCell(0, n[Y]/2, n[Z]/2)[c])
		ShiftMagL[c] = float64(sign)
		ShiftMagR[c] = -float64(sign)
	}

	sign := magsign(ShiftMagL[c])

	if mc < -tolerance {
		Shift(sign)
	} else if mc > tolerance {
		Shift(-sign)
	}
}

func CenterWallInRegion(region, magComp int) {
	PostStep(func() { centerWallInRegionProc(region, magComp) })
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
)

func getShiftSpeed() float64 {
	if lastShift != GetShiftPos() {
		lastV = (GetShiftPos() - lastShift) / (Time - lastT)
		lastShift = GetShiftPos()
		lastT = Time
	}
	return lastV
}

func GetDWxPos() float64 {
	M := &M
	mx := sAverageUniverse(M.Buffer().Comp(0))[0]
	c := Mesh().CellSize()
	n := Mesh().Size()
	position := mx * c[0] * float64(n[0]) / 2.
	return GetShiftPos() + position
}

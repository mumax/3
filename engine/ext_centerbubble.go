package engine

import (
	"github.com/mumax/3/data"
	"math"
)

func init() {
	DeclFunc("ext_centerBubble", CenterBubble, "centerBubble shifts m after each step to keep the bubble position close to the center of the window")
}

func centerBubble() {
	n := Mesh().Size()
	c := Mesh().CellSize()

	position := bubblePos()
	var centerIdx [2]int
	centerIdx[X] = int(math.Floor((position[X] - GetShiftPos()) / c[X]))
	centerIdx[Y] = int(math.Floor((position[Y] - GetShiftYPos()) / c[Y]))

	sign := magsign(M.GetCell(0, n[Y]/2, n[Z]/2)[Z]) //TODO make more robust with temperature?
	zero := data.Vector{0, 0, 0}
	if ShiftMagL == zero || ShiftMagR == zero || ShiftMagD == zero || ShiftMagU == zero {
		ShiftMagL[Z] = float64(sign)
		ShiftMagR[Z] = float64(sign)
		ShiftMagD[Z] = float64(sign)
		ShiftMagU[Z] = float64(sign)
	}

	//put bubble to center
	if centerIdx[X] != 0 {
		Shift(-centerIdx[X])
	}
	if centerIdx[Y] != 0 {
		YShift(-centerIdx[Y])
	}

}

// This post-step function centers the simulation window on a bubble
func CenterBubble() {
	PostStep(func() { centerBubble() })
}

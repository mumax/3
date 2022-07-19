package engine

import (
	"github.com/mumax/3/v3/data"
	"math"
)

func init() {
	DeclFunc("ext_centerBubble", CenterBubble, "centerBubble shifts m after each step to keep the bubble position close to the center of the window")
}

func centerBubble() {
	c := Mesh().CellSize()

	position := bubblePos()
	var centerIdx [2]int
	centerIdx[X] = int(math.Floor((position[X] - GetShiftPos()) / c[X]))
	centerIdx[Y] = int(math.Floor((position[Y] - GetShiftYPos()) / c[Y]))

	zero := data.Vector{0, 0, 0}
	if ShiftMagL == zero || ShiftMagR == zero || ShiftMagD == zero || ShiftMagU == zero {
		ShiftMagL[Z] = -BubbleMz
		ShiftMagR[Z] = -BubbleMz
		ShiftMagD[Z] = -BubbleMz
		ShiftMagU[Z] = -BubbleMz
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

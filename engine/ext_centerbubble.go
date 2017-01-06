package engine

import (
	"github.com/mumax/3/data"
)

func init() {
	DeclFunc("ext_centerBubble", CenterBubble, "centerBubble shifts m after each step to keep the bubble position close to the center of the window")
}

func centerBubble() {
	M := &M
	n := Mesh().Size()

	//TODO This part was copied from ext_bubblepos
	m := M.Buffer()
	mz := m.Comp(Z).HostCopy().Scalars()[0]
	posx, posy := 0, 0

	{
		max := float32(-1e32)
		for iy := range mz {
			var sum float32
			for ix := range mz[iy] {
				sum += mz[iy][ix]
			}
			if sum > max {
				posy = iy
				max = sum
			}
		}
	}

	{
		max := float32(-1e32)
		for ix := range mz[0] {
			var sum float32
			for iy := range mz {
				sum += mz[iy][ix]
			}
			if sum > max {
				posx = ix
				max = sum
			}
		}
	}

	zero := data.Vector{0, 0, 0}
	if ShiftMagL == zero || ShiftMagR == zero || ShiftMagD == zero || ShiftMagU == zero {
		sign := magsign(M.GetCell(0, n[Y]/2, n[Z]/2)[Z]) //TODO make more robust with temperature?
		ShiftMagL[Z] = float64(sign)
		ShiftMagR[Z] = float64(sign)
		ShiftMagD[Z] = float64(sign)
		ShiftMagU[Z] = float64(sign)
	}
	dx := n[X]/2 - posx
	dy := n[Y]/2 - posy

	//put bubble to center
	if dx != 0 {
		Shift(dx)
	}
	if dy != 0 {
		YShift(dy)
	}

}

// This post-step function centers the simulation window on a bubble
func CenterBubble() {
	PostStep(func() { centerBubble() })
}

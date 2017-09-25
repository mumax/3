package engine

import (
	"github.com/mumax/3/data"
	"math"
)

func init() {
	DeclFunc("ext_centerBubble", CenterBubble, "centerBubble shifts m after each step to keep the bubble position close to the center of the window")
}

func centerBubble() {
	M := &M
	n := Mesh().Size()

	m := M.Buffer()
	mz := m.Comp(Z).HostCopy().Scalars()[0]

	posx, posy := 0., 0.
	sign := magsign(M.GetCell(0, n[Y]/2, n[Z]/2)[Z]) //TODO make more robust with temperature?

	{
		var magsum float32
		var weightedsum float32

		for iy := range mz {
			for ix := range mz[0] {
				magsum += ((mz[iy][ix]*float32(-1*sign) + 1.) / 2.)
				weightedsum += ((mz[iy][ix]*float32(-1*sign) + 1.) / 2.) * float32(iy)
			}
		}
		posy = float64(weightedsum / magsum)
	}

	{
		var magsum float32
		var weightedsum float32

		for ix := range mz[0] {
			for iy := range mz {
				magsum += ((mz[iy][ix]*float32(-1*sign) + 1.) / 2.)
				weightedsum += ((mz[iy][ix]*float32(-1*sign) + 1.) / 2.) * float32(ix)
			}
		}
		posx = float64(weightedsum / magsum)
	}

	zero := data.Vector{0, 0, 0}
	if ShiftMagL == zero || ShiftMagR == zero || ShiftMagD == zero || ShiftMagU == zero {
		ShiftMagL[Z] = float64(sign)
		ShiftMagR[Z] = float64(sign)
		ShiftMagD[Z] = float64(sign)
		ShiftMagU[Z] = float64(sign)
	}
	dx := int(math.Floor(float64(n[X]/2) - posx))
	dy := int(math.Floor(float64(n[Y]/2) - posy))

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

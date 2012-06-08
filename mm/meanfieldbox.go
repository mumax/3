package mm

import (
	. "nimble-cube/nc"
)

// IDEA: Connect(box, "H_*") regex match.

type MeanFieldBox struct {
	M [3]<-chan []float32   "m"
	H [3][]chan<- []float32 "H"
}

func (box *MeanFieldBox) Run() {

	for {

		var mSum Vector

		for s := 0; s < NumWarp(); s++ {

			mSlice := Recv3(box.M)

			for i := range mSlice[X] {
				mSum[X] += mSlice[X][i]
				mSum[Y] += mSlice[Y][i]
				mSum[Z] += mSlice[Z][i]
			}
		}

		hx := mSum[X] * -0.01 / float32(N())
		hy := mSum[Y] * -0.04 / float32(N())
		hz := mSum[Z] * -0.95 / float32(N())

		for s := 0; s < NumWarp(); s++ {
			hSlice := Buffer3()
			Memset3(hSlice, Vector{hx, hy, hz})
			Send3(box.H, hSlice)
		}
	}
}

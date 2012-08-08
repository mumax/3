package mm

import (
	. "nimble-cube/nc"
)

// IDEA: Connect(box, "H_*") regex match.

type MeanFieldBox struct {
	M [3]<-chan Block   "m"
	H [3][]chan<- Block "H"
}

func (box *MeanFieldBox) Run() {

	for {

		var mSum Vector

		for s := 0; s < NumWarp(); s++ {

			mSlice := Recv3(box.M)

			for i := range mSlice[X].List {
				mSum[X] += mSlice[X].List[i]
				mSum[Y] += mSlice[Y].List[i]
				mSum[Z] += mSlice[Z].List[i]
			}

			Recycle3(mSlice)
		}

		hx := mSum[X] * -0.01 / float32(N())
		hy := mSum[Y] * -0.04 / float32(N())
		hz := mSum[Z] * -0.95 / float32(N())

		for s := 0; s < NumWarp(); s++ {
			hSlice := Buffer3()
			Memset(hSlice[X].List, hx)
			Memset(hSlice[Y].List, hy)
			Memset(hSlice[Z].List, hz) // TODO: memset on [3]Block
			Send3(box.H, hSlice)
		}
	}
}

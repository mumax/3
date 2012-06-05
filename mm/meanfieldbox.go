package mm

import (
	. "nimble-cube/nc"
)

type MeanFieldBox struct {
	m FanOut3
	h FanIn3
}

func (b *MeanFieldBox) Run() {

	for {

		var mSum Vector

		for s := 0; s < N/warp; s++ {
			mSlice := b.m.Recv()
			for i := range mSlice[X] {
				mSum[X] += mSlice[X][i]
				mSum[Y] += mSlice[Y][i]
				mSum[Z] += mSlice[Z][i]
			}
		}

		//	hx := mSum[X] * -0.01 / float32(N)
		//	hy := mSum[Y] * -0.02 / float32(N)
		//	hz := mSum[Z] * -0.97 / float32(N)

		for s := 0; s < N/warp; s++ {
			hSlice := Buffer3()
			//Memset3(hSlice, Vector{hx, hy, hz})
			Memset3(hSlice, Vector{0, 1, 0})
			b.h.Send(hSlice)
		}

	}
}

package mm

import (
	. "nimble-cube/nc"
)

type MeanFieldBox struct {
	m [3]<-chan []float32 // magnetization input
	h [3]chan<- []float32 // field output
}

func (box *MeanFieldBox) Run() {

	for {

		var mSum Vector
		var mSlice [3][]float32

		for s := 0; s < N/warp; s++ {

			mSlice[X] = <-box.m[X]
			mSlice[Y] = <-box.m[Y]
			mSlice[Z] = <-box.m[Z]

			for i := range mSlice[X] {
				mSum[X] += mSlice[X][i]
				mSum[Y] += mSlice[Y][i]
				mSum[Z] += mSlice[Z][i]
			}
		}

		hx := mSum[X] * -0.01 / float32(N)
		hy := mSum[Y] * -0.04 / float32(N)
		hz := mSum[Z] * -0.95 / float32(N)

		for s := 0; s < N/warp; s++ {
			hSlice := Buffer3()
			Memset3(hSlice, Vector{hx, hy, hz})
			Send3(box.h, hSlice)
		}
	}
}

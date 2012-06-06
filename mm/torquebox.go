package mm

import (
	. "nimble-cube/nc"
)

// Landau-Lifshitz torque.
type TorqueBox struct {
	m, h   [3]<-chan []float32 // input
	alpha  <-chan []float32    //input
	torque [3]chan<- []float32 // torque output
}

func (box *TorqueBox) Run() {
	for {
		mSlice := Recv3(box.m)
		hSlice := Recv3(box.h)
		aSlice := <-box.alpha
		tSlice := Buffer3()

		for i := range tSlice[X] {
			var m Vector
			var h Vector
			m[X], m[Y], m[Z] = mSlice[X][i], mSlice[Y][i], mSlice[Z][i]
			h[X], h[Y], h[Z] = hSlice[X][i], hSlice[Y][i], hSlice[Z][i]

			alpha := aSlice[i]

			mxh := m.Cross(h)
			t := mxh.Sub(m.Cross(mxh).Scaled(alpha))

			tSlice[X][i] = t[X]
			tSlice[Y][i] = t[Y]
			tSlice[Z][i] = t[Z]
		}
		Send3(box.torque, tSlice)
	}
}

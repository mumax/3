package mm

import (
	. "nimble-cube/nc"
)

// Landau-Lifshitz torque.
type LLGBox struct {
	M      [3]<-chan []float32   "m"
	H      [3]<-chan []float32   "H"
	Alpha  <-chan []float32      "alpha"
	Torque [3][]chan<- []float32 "torque"
}

func (box *LLGBox) Run() {
	for {
		mSlice := Recv3(box.M)
		hSlice := Recv3(box.H)
		aSlice := Recv(box.Alpha)
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
		Send3(box.Torque, tSlice)

		Recycle3(mSlice, hSlice)
		Recycle(aSlice)
	}
}

package mm

import (
	. "nimble-cube/nc"
)

// Landau-Lifshitz torque.
type LLGBox struct {
	M      [3]<-chan Block   "m"
	H      [3]<-chan Block   "H"
	Alpha  <-chan Block      "alpha"
	Torque [3][]chan<- Block "torque"
}

func (box *LLGBox) Run() {
	for {
		mSlice := Recv3(box.M)
		hSlice := Recv3(box.H)
		aSlice := Recv(box.Alpha)
		tSlice := Buffer3()

		for i := range tSlice[X].List {
			var m Vector
			var h Vector
			m[X], m[Y], m[Z] = mSlice[X].List[i], mSlice[Y].List[i], mSlice[Z].List[i]
			h[X], h[Y], h[Z] = hSlice[X].List[i], hSlice[Y].List[i], hSlice[Z].List[i]

			alpha := aSlice.List[i]

			mxh := m.Cross(h)
			t := mxh.Sub(m.Cross(mxh).Scaled(alpha))

			tSlice[X].List[i] = t[X]
			tSlice[Y].List[i] = t[Y]
			tSlice[Z].List[i] = t[Z]
		}
		Send3(box.Torque, tSlice)

		Recycle3(mSlice, hSlice)
		Recycle(aSlice)
	}
}

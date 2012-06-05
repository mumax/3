package mm

import (
	. "nimble-cube/nc"
)

type TorqueBox struct {
	m, h FanOut3
	t    FanIn3
}

func (box *TorqueBox) Run() {
	for {
		mSlice := box.m.Recv()
		hSlice := box.h.Recv()
		tSlice := Buffer3()

		for i := range tSlice[X] {
			var m Vector
			var h Vector
			m[X], m[Y], m[Z] = mSlice[X][i], mSlice[Y][i], mSlice[Z][i]
			h[X], h[Y], h[Z] = hSlice[X][i], hSlice[Y][i], hSlice[Z][i]

			mxh := m.Cross(h)
			t := mxh //.Sub(m.Cross(mxh).Scale(alpha))

			tSlice[X][i] = t[X]
			tSlice[Y][i] = t[Y]
			tSlice[Z][i] = t[Z]
		}
		box.t.Send(tSlice)
	}
}

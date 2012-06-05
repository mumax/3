package mm

import (
	. "nimble-cube/nc"
)

type TorqueBox struct {
	m, h FanOut3
	t    FanIn3
}

func (b *TorqueBox) Run() {
	for {
		mSlice := b.m.Recv()
		hSlice := b.h.Recv()
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
		b.t.Send(tSlice)
	}
}

//func RunTorque(tick <-chan float32, torque VecChan, mRecv VecRecv, hRecv VecRecv, alphaRecv Recv) {
//
//	// loop over blocks
//	for I := 0; I < N; I += warp {
//
//		torqueSlice := VecBuffer()
//		//alphaList := alphaChan.Recv()
//		mList := mRecv.Recv()
//		hList := hRecv.Recv()
//		_ = alphaRecv.Recv()
//
//		// loop inside blocks
//		for i := range torqueSlice[X] {
//			var m Vector
//			var h Vector
//			m[X], m[Y], m[Z] = mList[X][i], mList[Y][i], mList[Z][i]
//			h[X], h[Y], h[Z] = hList[X][i], hList[Y][i], hList[Z][i]
//			//alpha := alphaList[i]
//
//			mxh := m.Cross(h)
//			tq := mxh //.Sub(m.Cross(mxh).Scale(alpha))
//
//			torqueSlice[X][i] = tq[X]
//			torqueSlice[Y][i] = tq[Y]
//			torqueSlice[Z][i] = tq[Z]
//		}
//		torqueChan.Send(torqueSlice)
//	}
//}

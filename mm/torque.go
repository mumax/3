package mm

import (
	. "nimble-cube/nc"
)

func RunTorque(tick <-chan float32, torque VecChan, mRecv VecRecv, hRecv VecRecv, alphaRecv Recv) {

	// loop over blocks
	for I := 0; I < N; I += warp {

		torqueSlice := VecBuffer()
		//alphaList := alphaChan.Recv()
		mList := mRecv.Recv()
		hList := hRecv.Recv()
		_ = alphaRecv.Recv()

		// loop inside blocks
		for i := range torqueSlice[X] {
			var m Vector
			var h Vector
			m[X], m[Y], m[Z] = mList[X][i], mList[Y][i], mList[Z][i]
			h[X], h[Y], h[Z] = hList[X][i], hList[Y][i], hList[Z][i]
			//alpha := alphaList[i]

			mxh := m.Cross(h)
			tq := mxh //.Sub(m.Cross(mxh).Scale(alpha))

			torqueSlice[X][i] = tq[X]
			torqueSlice[Y][i] = tq[Y]
			torqueSlice[Z][i] = tq[Z]
		}
		torqueChan.Send(torqueSlice)
	}
}

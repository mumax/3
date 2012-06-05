package mm

import (
	. "nimble-cube/nc"
)

type EulerBox struct {
	m   FanIn3
	mIn FanOut3
	t   FanOut3
	dt  float32
}

// m0: initial value, to be overwritten by result when done.
func (b *EulerBox) Run(m0 [3][]float32, steps int) {

	// Fan-out input m buffer if not yet done so.
	if b.mIn.IsNil() {
		b.mIn = b.m.FanOut(DefaultBufSize())
	}

	// send initial value m0 down the m pipe
	for I := 0; I < N; I += warp {
		b.m.Send([3][]float32{m0[X][I : I+warp], m0[Y][I : I+warp], m0[Z][I : I+warp]})
	}

	//TODO: select over components?
	for s := 0; s < steps; s++ {
		for I := 0; I < N; I += warp {

			m0Slice := b.mIn.Recv()
			tSlice := b.t.Recv()
			m1Slice := Buffer3()

			for i := range m1Slice[X] {
				var m1 Vector
				m1[X] = m0Slice[X][i] + b.dt*tSlice[X][i]
				m1[Y] = m0Slice[Y][i] + b.dt*tSlice[Y][i]
				m1[Z] = m0Slice[Z][i] + b.dt*tSlice[Z][i]
				m1 = m1.Normalized()
				m1Slice[X][i] = m1[X]
				m1Slice[Y][i] = m1[Y]
				m1Slice[Z][i] = m1[Z]
			}

			if s < steps-1 {
				b.m.Send(m1Slice)
			} else {
				copy(m0[X][I:I+warp], m1Slice[X])
				copy(m0[Y][I:I+warp], m1Slice[Y])
				copy(m0[Z][I:I+warp], m1Slice[Z])
				//RECYCLE
			}
		}
	}
}

package mm

import (
	. "nimble-cube/nc"
)

// Euler solver.
type EulerBox struct {
	mOut   [3][]chan<- []float32 // magnetization, output
	time   []chan<- float64      // time, output
	step   []chan<- float64      // time step, output
	torque [3]<-chan []float32   // torque, input
	mIn    [3]<-chan []float32   // AUTOMATICALLY SET: magnetization input ??
	t      float64               // local copy of time
	dt     float32               // local copy of time step
	steps  int                   // local copy of total time steps
}

// m0: initial value, to be overwritten by result when done.
func (box *EulerBox) Run(m0 [3][]float32, steps int) {

	// Fan-out input m buffer if not yet done so.
	//	if box.mIn==nil{
	//		box.mIn = box.m.FanOut(DefaultBufSize())
	//	}

	// send initial value m0 down the m pipe
	for I := 0; I < N; I += warp {
		m0Slice := [3][]float32{m0[X][I : I+warp], m0[Y][I : I+warp], m0[Z][I : I+warp]}
		Send3(box.mOut, m0Slice)
	}

	for s := 0; s < steps; s++ {
		// Send time first, so others can prepare my input.
		SendFloat64(box.time, box.t)
		SendFloat64(box.step, float64(box.steps))

		for I := 0; I < N; I += warp {

			m0Slice := Recv3(box.mIn)
			tSlice := Recv3(box.torque)
			m1Slice := Buffer3()

			for i := range m1Slice[X] {
				var m1 Vector
				m1[X] = m0Slice[X][i] + box.dt*tSlice[X][i]
				m1[Y] = m0Slice[Y][i] + box.dt*tSlice[Y][i]
				m1[Z] = m0Slice[Z][i] + box.dt*tSlice[Z][i]
				m1 = m1.Normalized()
				m1Slice[X][i] = m1[X]
				m1Slice[Y][i] = m1[Y]
				m1Slice[Z][i] = m1[Z]
			}

			if s < steps-1 {
				Send3(box.mOut, m1Slice)
			} else {
				copy(m0[X][I:I+warp], m1Slice[X])
				copy(m0[Y][I:I+warp], m1Slice[Y])
				copy(m0[Z][I:I+warp], m1Slice[Z])
				//RECYCLE
			}
		}
		box.t += (float64(box.dt))
		box.steps++
	}
}

package mm

import (
	. "nimble-cube/nc"
)

// Euler solver.
type EulerBox struct {
	MOut   [3][]chan<- Block "m"
	Time   []chan<- float64  "time"
	Step   []chan<- float64  "step"
	Torque [3]<-chan Block   "torque"
	MIn    [3]<-chan Block   "m"
	t      float64           // local copy of time
	dt     float32           // local copy of time step
	steps  int               // local copy of total time steps
}

// m0: initial value, to be overwritten by result when done.
func (box *EulerBox) Run(m0 [3]Block, steps int) {

	// send initial value m0 down the m pipe
	for s := 0; s < NumWarp(); s++ {
		m0Slice := [3]Block{
			m0[X].Slice(s),
			m0[Y].Slice(s),
			m0[Z].Slice(s)}
		Send3(box.MOut, m0Slice)
	}

	for s := 0; s < steps; s++ {
		// Send time first, so others can prepare my input.
		SendFloat64(box.Time, box.t)
		SendFloat64(box.Step, float64(box.steps))

		for I := 0; I < N(); I += WarpLen() {

			m0Slice := Recv3(box.MIn)
			tSlice := Recv3(box.Torque)
			m1Slice := Buffer3()

			for i := range m1Slice[X].List {
				var m1 Vector
				m1[X] = m0Slice[X].List[i] + box.dt*tSlice[X].List[i]
				m1[Y] = m0Slice[Y].List[i] + box.dt*tSlice[Y].List[i]
				m1[Z] = m0Slice[Z].List[i] + box.dt*tSlice[Z].List[i]
				m1 = m1.Normalized()
				m1Slice[X].List[i] = m1[X]
				m1Slice[Y].List[i] = m1[Y]
				m1Slice[Z].List[i] = m1[Z]
			}

			if s < steps-1 {
				// still need to step
				Send3(box.MOut, m1Slice)
			} else {
				// done stepping, copy result back to m0
				copy(m0[X].List[I:I+WarpLen()], m1Slice[X].List)
				copy(m0[Y].List[I:I+WarpLen()], m1Slice[Y].List)
				copy(m0[Z].List[I:I+WarpLen()], m1Slice[Z].List)
				Recycle3(m1Slice)
			}
			Recycle3(m0Slice, tSlice)
		}
		box.t += (float64(box.dt))
		box.steps++
	}
}

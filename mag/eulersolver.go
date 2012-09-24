package mag

import (
	"math"
	"nimble-cube/core"
)



func EulerStep(m, torque [3][]float32, dt float64) {
	h := float32(dt)
	for i := range m[0] {
		mx := m[0][i] + h*torque[0][i]
		my := m[1][i] + h*torque[1][i]
		mz := m[2][i] + h*torque[2][i]
		norm := float32(math.Sqrt(float64(mx*mx + my*my + mz*mz)))
		if norm != 0 {
			m[0][i] = mx / norm
			m[1][i] = my / norm
			m[2][i] = mz / norm
		}
	}
}

// Euler solver.
type Euler struct {
	y1 core.Chan3
	dy core.RChan3
	dt float32
	blocklen int
}


func NewEuler(y2 core.Chan3, dy core.RChan3, dt float32) *EulerBox {
	return &Euler{y2, y1, dy, dt, core.BlockLen(y2.Size())}
}

func (e *Euler) Steps(steps int) {
	n := core.Prod(e.y2.Size())
	block := e.blocklen


---
	lock torque for reading
	lock m for writing (implies safe reading as well)
	overwrite m += torque*dt
---

	// write y2 once
	
	for s := 0; s < steps; s++ {

		for I:=0; i<n; i+=block{

				y2 := e.y2.WriteNext(block)
				y1 := e.y
	
			}
	}
}



			//	var m1 Vector
			//	m1[X] = box.M[X][i] + float32(box.Dt)*box.Torque[X][i]
			//	m1[Y] = box.M[Y][i] + float32(box.Dt)*box.Torque[Y][i]
			//	m1[Z] = box.M[Z][i] + float32(box.Dt)*box.Torque[Z][i]
			//	m1 = m1.Normalized()
			//	box.M[X][i] = m1[X]
			//	box.M[Y][i] = m1[Y]
			//	box.M[Z][i] = m1[Z]

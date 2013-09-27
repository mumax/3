package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
	"math"
)

func HeunStep(e *solver) {
	y := e.y
	dy0 := cuda.Buffer(3, e.y.Mesh())
	defer cuda.Recycle(dy0)

	dt := float32(e.Dt_si * e.dt_mul) // could check here if it is in float32 ranges
	util.Assert(dt > 0)

	// stage 1
	{
		e.torqueFn(dy0)
		e.NEval++
		cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy
	}

	// stage 2
	{
		dy := cuda.Buffer(3, e.y.Mesh())
		defer cuda.Recycle(dy)
		*e.time += e.Dt_si
		e.torqueFn(dy)
		e.NEval++

		err := 0.0
		if e.FixDt == 0 { // time step not fixed
			err = cuda.MaxVecDiff(dy0, dy) * float64(dt)
		}

		if err < e.MaxErr || e.Dt_si <= e.MinDt { // mindt check to avoid infinite loop
			// step OK
			cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
			e.postStep(y)
			e.NSteps++
			e.adaptDt(math.Pow(e.MaxErr/err, 1./2.))
			e.LastErr = err
		} else {
			// undo bad step
			util.Assert(e.FixDt == 0)
			*e.time -= e.Dt_si
			cuda.Madd2(y, y, dy0, 1, -dt)
			e.NUndone++
			e.adaptDt(math.Pow(e.MaxErr/err, 1./3.))
		}
	}
}

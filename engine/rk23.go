package engine

/*
import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)


// k1 = f(tn, yn)
// k2 = f(tn + 1/2 h, yn + 1/2 h k1)
// k3 = f(tn + 3/4 h, yn + 3/4 h k2)
// y{n+1}  = yn + 2/9 h k1 + 1/3 h k2 + 4/9 h k3
// k4 = f(tn + h, y{n+1})
// z{n+1} = yn + 7/24 h k1 + 1/4 h k2 + 1/3 h k3 + 1/8 h k4

func RK23Step(s *solver, y *data.Slice) {

	h := float32(s.Dt_si * *s.dt_mul)
	util.Assert(dt > 0)
	N := y.Size()

	k1 := cuda.Buffer(VECTOR, N)
	defer cuda.Recycle(k1)

	y2 := cuda.Buffer(VECTOR, N)
	defer cuda.Recycle(y2)
	k2 := cuda.Buffer(VECTOR, N)
	defer cuda.Recycle(k2)

	y3 := cuda.Buffer(VECTOR, N)
	defer cuda.Recycle(y3)
	k3 := cuda.Buffer(VECTOR, N)
	defer cuda.Recycle(k3)


	// stage 1
	t0 := Time
	s.torqueFn(k1);  s.NEval++

	// stage 2
	cuda.Madd2(y2, y, k1, 1, 0.5 * h) // y2 = 1 y + (0.5 * h) k1
	Time = t0 + 0.5 * h
	s.torqueFn(k2); s.NEval++

	// stage 3
	cuda.Madd2(y3, y, k2, 1, 3./4. * h) // y3 = 1 y + (3/4 * h) k2
	Time = t0 + 3./4. * h
	s.torqueFn(k3); s.NEval++

	// low-order torque
	kLow := cuda.Buffer(VECTOR, N)
	defer cuda.Recycle(kLow)
	cuda.Madd3(kLow, k1, k2, k3, 2./9., 1./3., 4./9)
	yLow := cuda.Buffer(VECTOR, N)
	defer cuda.Recycle(yLow)
	cuda.Madd3(yLow, y


	k4 := cuda.Buffer(VECTOR, N)
	defer cuda.Recycle(k4)


	// stage 2
	dy := cuda.Buffer(3, y.Size())
	defer cuda.Recycle(dy)
	Time += s.Dt_si
	s.torqueFn(dy)
	s.NEval++

	// determine error
	err := 0.0
	if s.FixDt == 0 { // time step not fixed
		err = cuda.MaxVecDiff(dy0, dy) * float64(dt)
	}

	// adjust next time step
	if err < s.MaxErr || s.Dt_si <= s.MinDt { // mindt check to avoid infinite loop
		// step OK
		cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
		s.postStep()
		s.NSteps++
		s.adaptDt(math.Pow(s.MaxErr/err, 1./2.))
		s.LastErr = err
	} else {
		// undo bad step
		util.Assert(s.FixDt == 0)
		Time -= s.Dt_si
		cuda.Madd2(y, y, dy0, 1, -dt)
		s.NUndone++
		s.adaptDt(math.Pow(s.MaxErr/err, 1./3.))
	}

}
*/

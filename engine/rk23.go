package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

// Bogacki-Shampine solver. 3rd order, 3 evaluations per step, adaptive step.
// 	http://en.wikipedia.org/wiki/Bogacki-Shampine_method
//
// 	k1 = f(tn, yn)
// 	k2 = f(tn + 1/2 h, yn + 1/2 h k1)
// 	k3 = f(tn + 3/4 h, yn + 3/4 h k2)
// 	y{n+1}  = yn + 2/9 h k1 + 1/3 h k2 + 4/9 h k3            // 3rd order
// 	k4 = f(tn + h, y{n+1})
// 	z{n+1} = yn + 7/24 h k1 + 1/4 h k2 + 1/3 h k3 + 1/8 h k4 // 2nd order

type RK23 struct {
	k1 *data.Slice // torque at end of step is kept for beginning of next step
}

func (rk *RK23) Step(unused *data.Slice) {
	m := M.Buffer()
	size := m.Size()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	// first step ever: one-time k1 init and eval
	if rk.k1 == nil {
		rk.k1 = cuda.NewSlice(3, size)
		torqueFn(rk.k1)
	}

	t0 := Time
	// backup magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	k1 := rk.k1 // from previous step
	k2, k3, k4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)

	h := float32(Dt_si * *dt_mul) // internal time step = Dt * gammaLL

	// there is no explicit stage 1: k1 from previous step

	// stage 2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(m, m0, k1, 1, (1./2.)*h) // m = m0*1 + k1*h/2
	//postStep()
	torqueFn(k2)

	// stage 3
	Time = t0 + (3./4.)*Dt_si
	cuda.Madd2(m, m0, k2, 1, (3./4.)*h) // m = m0*1 + k2*3/4
	//postStep()
	torqueFn(k3)

	// 3rd order solution
	τ3 := cuda.Buffer(3, size)
	defer cuda.Recycle(τ3)
	cuda.Madd3(τ3, k1, k2, k3, (2. / 9.), (1. / 3.), (4. / 9.))
	cuda.Madd2(m, m0, τ3, 1, h)
	solverPostStep()

	// error estimate
	τ2 := cuda.Buffer(3, size)
	defer cuda.Recycle(τ2)
	Time = t0 + Dt_si
	torqueFn(k4)
	madd4(τ2, k1, k2, k3, k4, (7. / 24.), (1. / 4.), (1. / 3.), (1. / 8.))

	// determine error
	err := 0.0
	if FixDt == 0 { // time step not fixed
		err = cuda.MaxVecDiff(τ2, τ3) * float64(h)
	}

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt { // mindt check to avoid infinite loop
		// step OK
		LastErr = err // output error/step for good steps only
		NSteps++
		Time = t0 + Dt_si
		adaptDt(math.Pow(MaxErr/err, 1./3.))
		data.Copy(rk.k1, k4) // FSAL
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m, m0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./4.))
	}
}

// TODO: into cuda
func madd4(dst, src1, src2, src3, src4 *data.Slice, w1, w2, w3, w4 float32) {
	cuda.Madd3(dst, src1, src2, src3, w1, w2, w3)
	cuda.Madd2(dst, dst, src4, 1, w4)
}

//func RK23Step(s *solver, y *data.Slice) {
//
//	h := float32(s.Dt_si * *s.dt_mul)
//	util.Assert(dt > 0)
//	N := y.Size()
//
//	k1 := cuda.Buffer(VECTOR, N)
//	defer cuda.Recycle(k1)
//
//	y2 := cuda.Buffer(VECTOR, N)
//	defer cuda.Recycle(y2)
//	k2 := cuda.Buffer(VECTOR, N)
//	defer cuda.Recycle(k2)
//
//	y3 := cuda.Buffer(VECTOR, N)
//	defer cuda.Recycle(y3)
//	k3 := cuda.Buffer(VECTOR, N)
//	defer cuda.Recycle(k3)
//
//
//	// stage 1
//	t0 := Time
//	s.torqueFn(k1);  s.NEval++
//
//	// stage 2
//	cuda.Madd2(y2, y, k1, 1, 0.5 * h) // y2 = 1 y + (0.5 * h) k1
//	Time = t0 + 0.5 * h
//	s.torqueFn(k2); s.NEval++
//
//	// stage 3
//	cuda.Madd2(y3, y, k2, 1, 3./4. * h) // y3 = 1 y + (3/4 * h) k2
//	Time = t0 + 3./4. * h
//	s.torqueFn(k3); s.NEval++
//
//	// low-order torque
//	kLow := cuda.Buffer(VECTOR, N)
//	defer cuda.Recycle(kLow)
//	cuda.Madd3(kLow, k1, k2, k3, 2./9., 1./3., 4./9)
//	yLow := cuda.Buffer(VECTOR, N)
//	defer cuda.Recycle(yLow)
//	cuda.Madd3(yLow, y
//
//
//	k4 := cuda.Buffer(VECTOR, N)
//	defer cuda.Recycle(k4)
//
//
//	// stage 2
//	dy := cuda.Buffer(3, y.Size())
//	defer cuda.Recycle(dy)
//	Time += s.Dt_si
//	s.torqueFn(dy)
//	s.NEval++
//
//	// determine error
//	err := 0.0
//	if s.FixDt == 0 { // time step not fixed
//		err = cuda.MaxVecDiff(dy0, dy) * float64(dt)
//	}
//
//	// adjust next time step
//	if err < s.MaxErr || s.Dt_si <= s.MinDt { // mindt check to avoid infinite loop
//		// step OK
//		cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
//		s.postStep()
//		s.NSteps++
//		s.adaptDt(math.Pow(s.MaxErr/err, 1./2.))
//		s.LastErr = err
//	} else {
//		// undo bad step
//		util.Assert(s.FixDt == 0)
//		Time -= s.Dt_si
//		cuda.Madd2(y, y, dy0, 1, -dt)
//		s.NUndone++
//		s.adaptDt(math.Pow(s.MaxErr/err, 1./3.))
//	}
//
//}

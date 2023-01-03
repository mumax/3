package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

type RK56 struct {
	k1 *data.Slice // torque at end of step is kept for beginning of next step
}

func (rk *RK56) Step() {

	m := M.Buffer()
	size := m.Size()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	// upon resize: remove wrongly sized k1
	if rk.k1.Size() != m.Size() {
		rk.Free()
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

	k2, k3, k4, k5, k6, k7, k8 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)
	defer cuda.Recycle(k5)
	defer cuda.Recycle(k6)
	defer cuda.Recycle(k7)
	defer cuda.Recycle(k8)
	//k2 will be recyled as k9

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL

	// stage 1
	torqueFn(rk.k1)

	// stage 2
	Time = t0 + (1./6.)*Dt_si
	cuda.Madd2(m, m, rk.k1, 1, (1./6.)*h) // m = m*1 + k1*h/6
	M.normalize()
	torqueFn(k2)

	// stage 3
	Time = t0 + (4./15.)*Dt_si
	cuda.Madd3(m, m0, rk.k1, k2, 1, (4./75.)*h, (16./75.)*h)
	M.normalize()
	torqueFn(k3)

	// stage 4
	Time = t0 + (2./3.)*Dt_si
	cuda.Madd4(m, m0, rk.k1, k2, k3, 1, (5./6.)*h, (-8./3.)*h, (5./2.)*h)
	M.normalize()
	torqueFn(k4)

	// stage 5
	Time = t0 + (4./5.)*Dt_si
	cuda.Madd5(m, m0, rk.k1, k2, k3, k4, 1, (-8./5.)*h, (144./25.)*h, (-4.)*h, (16./25.)*h)
	M.normalize()
	torqueFn(k5)

	// stage 6
	Time = t0 + (1.)*Dt_si
	cuda.Madd6(m, m0, rk.k1, k2, k3, k4, k5, 1, (361./320.)*h, (-18./5.)*h, (407./128.)*h, (-11./80.)*h, (55./128.)*h)
	M.normalize()
	torqueFn(k6)

	// stage 7
	Time = t0
	cuda.Madd5(m, m0, rk.k1, k3, k4, k5, 1, (-11./640.)*h, (11./256.)*h, (-11/160.)*h, (11./256.)*h)
	M.normalize()
	torqueFn(k7)

	// stage 8
	Time = t0 + (1.)*Dt_si
	cuda.Madd7(m, m0, rk.k1, k2, k3, k4, k5, k7, 1, (93./640.)*h, (-18./5.)*h, (803./256.)*h, (-11./160.)*h, (99./256.)*h, (1.)*h)
	M.normalize()
	torqueFn(k8)

	// stage 9: 6th order solution
	Time = t0 + (1.)*Dt_si
	//madd6(m, m0, k1, k3, k4, k5, k6, 1, (31./384.)*h, (1125./2816.)*h, (9./32.)*h, (125./768.)*h, (5./66.)*h)
	cuda.Madd7(m, m0, rk.k1, k3, k4, k5, k7, k8, 1, (7./1408.)*h, (1125./2816.)*h, (9./32.)*h, (125./768.)*h, (5./66.)*h, (5./66.)*h)
	M.normalize()
	torqueFn(k2) // re-use k2

	// error estimate
	Err := cuda.Buffer(3, size)
	defer cuda.Recycle(Err)
	cuda.Madd4(Err, rk.k1, k6, k7, k8, (-5. / 66.), (-5. / 66.), (5. / 66.), (5. / 66.))

	// determine error
	err := cuda.MaxVecNorm(Err) * float64(h)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		//Passed absolute error. Check relative error...
		errnorm := cuda.Buffer(1, size)
		defer cuda.Recycle(errnorm)
		cuda.VecNorm(errnorm, Err)
		ddtnorm := cuda.Buffer(1, size)
		defer cuda.Recycle(ddtnorm)
		cuda.VecNorm(ddtnorm, k2)
		maxdm := cuda.MaxVecNorm(k2)
		fail := 0
		rlerr := float64(0.0)
		if maxdm < MinSlope { // Only step using relerr if dmdt is big enough. Overcomes equilibrium problem
			fail = 0
		} else {
			cuda.Div(errnorm, errnorm, ddtnorm) //re-use errnorm
			rlerr = float64(cuda.MaxAbs(errnorm))
			fail = 1
		}
		if fail == 0 || RelErr <= 0.0 || rlerr < RelErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
			// step OK
			setLastErr(err)
			setMaxTorque(k2)
			NSteps++
			Time = t0 + Dt_si
			if fail == 0 {
				adaptDt(math.Pow(MaxErr/err, 1./6.))
			} else {
				adaptDt(math.Pow(RelErr/rlerr, 1./6.))
			}
			data.Copy(rk.k1, k2) // FSAL
		} else {
			// undo bad step
			//util.Println("Bad step at t=", t0, ", err=", err)
			util.Assert(FixDt == 0)
			Time = t0
			data.Copy(m, m0)
			NUndone++
			adaptDt(math.Pow(RelErr/rlerr, 1./7.))
		}
	} else {
		// undo bad step
		//util.Println("Bad step at t=", t0, ", err=", err)
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m, m0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./7.))
	}
}

func (rk *RK56) Free() {
	rk.k1.Free()
	rk.k1 = nil
}

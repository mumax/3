package engine

import (
	"github.com/mumax/3/v3/cuda"
	"github.com/mumax/3/v3/data"
	"github.com/mumax/3/v3/util"
	"math"
)

// Classical 4th order RK solver.
type RK4 struct {
}

func (rk *RK4) Step() {
	m := M.Buffer()
	size := m.Size()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	t0 := Time
	// backup magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	k1, k2, k3, k4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	defer cuda.Recycle(k1)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL

	// stage 1
	torqueFn(k1)

	// stage 2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(m, m, k1, 1, (1./2.)*h) // m = m*1 + k1*h/2
	M.normalize()
	torqueFn(k2)

	// stage 3
	cuda.Madd2(m, m0, k2, 1, (1./2.)*h) // m = m0*1 + k2*1/2
	M.normalize()
	torqueFn(k3)

	// stage 4
	Time = t0 + Dt_si
	cuda.Madd2(m, m0, k3, 1, 1.*h) // m = m0*1 + k3*1
	M.normalize()
	torqueFn(k4)

	err := cuda.MaxVecDiff(k1, k4) * float64(h)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		// 4th order solution
		cuda.Madd5(m, m0, k1, k2, k3, k4, 1, (1./6.)*h, (1./3.)*h, (1./3.)*h, (1./6.)*h)
		M.normalize()
		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./4.))
		setLastErr(err)
		setMaxTorque(k4)
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m, m0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./5.))
	}
}

func (_ *RK4) Free() {}

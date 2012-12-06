package uni

//// This file implements a plethora of Runge-Kutta methods
//// Coefficients and descriptions are taken from wikipedia.
////
//// TODO: for the error estimate, do not make the difference
//// between m_est and m_accurate, but between k_est and k_accurate.
//// this avoids truncation errors when subtracting two nearly equal
//// quantities.
////
//// TODO: perhaps all the coefficients should be carefully
//// double-checked for typos?
////
//import (
//	"code.google.com/p/nimble-cube/nimble"
//	"fmt"
//	"math"
//)
//
//// General Runge-Kutta solver
////
//// 	y_(n+1) = y_(n) + dt * sum b_i * k_i
////  k_i = torque(m0 + dt * sum a_ij * k_j, t0 + c_i*dt)
////
//// butcher tableau:
////
//// 	c0 | a00 a01 ...
//// 	c1 | a10 a11 ...
//// 	...| ... ... ...
//// 	----------------
//// 	   | b0  b1  ...
//// 	   | b2_0 b2_1 ...
////
//type RK struct {
//	stages         int
//	fsal           bool    // First Same as Last?
//	fsal_initiated bool    //
//	errororder     float64 // the order of the less acurate solution used for the error estimate
//	a              [][]float32
//	b              []float32
//	c              []float32
//	b2             []float32       // weights to get lower order solution for error estimate, may be nil
//	k              []nimble.Slice //
//	m0             nimble.Slice   // buffer to backup the starting magnetization
//	mbackup        nimble.Slice   // backup to undo bad steps TODO: can some other bufer be re-used for this?
//}
//
//// rk1: Euler's method
//// 0 | 0
//// -----
////   | 1
//func NewRK1() *RK {
//	rk := newRK(1)
//	rk.b[0] = 1.
//	return rk
//}
//
//// rk2: Heun's method
//// 0 | 0  0
//// 1 | 1  0
//// --------
////   |.5 .5
//func NewRK2() *RK {
//	rk := newRK(2)
//	rk.c[1] = 1.
//	rk.a[1][0] = 1.
//	rk.b[0] = .5
//	rk.b[1] = .5
//	return rk
//}
//
//// rk12: Adaptive Heun
//// 0 | 0  0
//// 1 | 1  0
//// --------
////   |.5 .5
////   | 1  0
//func NewRK12() *RK {
//	rk := NewRK2()
//	rk.initAdaptive(1.)
//	rk.b2 = []float32{1., 0.}
//	return rk
//}
//
//// rk3: Kutta's method
////  0  | 0    0  0
////  1/2| 1/2  0  0
////  1  | -1   2  0
//// ----------------
////     | 1/6 2/3 1/6
//func NewRK3() *RK {
//	rk := newRK(3)
//	rk.c = []float32{0., 1. / 2., 1.}
//	rk.a = [][]float32{
//		{0., 0., 0.},
//		{1. / 2., 0., 0.},
//		{-1., 2., 0}}
//	rk.b = []float32{1. / 6., 2. / 3., 1. / 6.}
//	return rk
//}
//
//// rk23: Bogacki–Shampine method
//// The Bogacki–Shampine method is a Runge–Kutta method of order three
//// with four stages with the First Same As Last (FSAL) property,
//// so that it uses approximately three function evaluations per step.
//// It has an embedded second-order method
//// which can be used to implement adaptive step size.
//func NewRK23() *RK {
//	rk := newRK(4)
//	rk.fsal = true
//	rk.c = []float32{0., 1. / 2., 3. / 4., 1.}
//	rk.a = [][]float32{
//		{0., 0., 0., 0.},
//		{1. / 2., 0., 0., 0.},
//		{0., 3. / 4., 0., 0.},
//		{2. / 9., 1. / 3., 4. / 9., 0.}}
//	rk.b = []float32{2. / 9., 1. / 3., 4. / 9., 0.}
//	rk.initAdaptive(2.)
//	rk.b2 = []float32{7. / 24, 1. / 4., 1. / 3., 1. / 8.}
//	return rk
//}
//
//// rk4: The classical Runge-Kutta method
////  0  | 0    0  0   0
////  1/2| 1/2  0  0   0
////  1/2| 0  1/2  0   0
////  1  | 0    0  1   0
//// ---------------------
////     | 1/6 1/3 1/3 1/6
//func NewRK4() *RK {
//	rk := newRK(4)
//	rk.c = []float32{0., 1. / 2., 1. / 2., 1.}
//	rk.a = [][]float32{
//		{0., 0., 0., 0.},
//		{1. / 2., 0., 0., 0.},
//		{0., 1. / 2., 0., 0.},
//		{0., 0., 1., 0}}
//	rk.b = []float32{1. / 6., 1. / 3., 1. / 3., 1. / 6.}
//	return rk
//}
//
//// Cash-Karp method
//// Uses six function evaluations to calculate fourth- and fifth-order accurate solutions.
//func NewRKCK() *RK {
//	rk := newRK(6)
//	rk.c = []float32{0., 1. / 5., 3. / 10., 3. / 5., 1., 7. / 8.}
//	rk.a = [][]float32{
//		{0, 0, 0, 0, 0, 0},
//		{1. / 5., 0, 0, 0, 0, 0},
//		{3. / 40., 9. / 40., 0, 0, 0, 0},
//		{3. / 10., -9. / 10., 6. / 5., 0, 0, 0},
//		{-11. / 54., 5. / 2., -70. / 27., 35. / 27., 0, 0},
//		{1631. / 55296., 175. / 512., 575. / 13824., 44275. / 110592., 253. / 4096., 0.}}
//	rk.b = []float32{37. / 378., 0, 250. / 621., 125. / 594., 0, 512. / 1771.}
//	rk.initAdaptive(4.)
//	rk.b2 = []float32{2825. / 27648., 0, 18575. / 48384., 13525. / 55296., 277. / 14336., 1. / 4.}
//	return rk
//}
//
//// Dormand-Prince method
////
//// Uses six function evaluations to calculate fourth- and fifth-order accurate solutions.
//// The Dormand–Prince method has seven stages, but it uses only six function evaluations
//// per step because it has the FSAL (First Same As Last) property: the last stage is
//// evaluated at the same point as the first stage of the next step.
//// Dormand and Prince choose the coefficients of their method to minimize the error of
//// the fifth-order solution. This is the main difference with the Fehlberg method,
//// which was constructed so that the fourth-order solution has a small error.
//// For this reason, the Dormand–Prince method is more suitable when the higher-order
//// solution is used to continue the integration. [wikipedia.org]
////
//// TODO: check if b2-b are not swapped.
//// The final 0. in the 5th-order solution seems suspicious,
//// but it's like that on wikipedia.
//func NewRKDP() *RK {
//	rk := newRK(7)
//	rk.fsal = true
//	rk.c = []float32{0., 1. / 5., 3. / 10., 4. / 5., 8. / 9., 1., 1.}
//	rk.a = [][]float32{
//		{0., 0., 0., 0., 0., 0., 0.},
//		{1. / 5., 0., 0., 0., 0., 0., 0.},
//		{3. / 40., 9. / 40., 0., 0., 0., 0., 0.},
//		{44. / 45., -56. / 15., 32. / 9., 0., 0., 0., 0.},
//		{19372. / 6561., -25360. / 2187., 64448. / 6561., -212. / 729., 0., 0., 0.},
//		{9017. / 3168., -355. / 33., 46732. / 5247., 49. / 176., -5103. / 18656., 0., 0.},
//		{35. / 384., 0., 500. / 1113., 125. / 192., -2187. / 6784., 11. / 84., 0., 0.}}
//	rk.b = []float32{35. / 384., 0., 500. / 1113., 125. / 192., -2187. / 6784., 11. / 84., 0}
//	rk.initAdaptive(4.)
//	rk.b2 = []float32{5179. / 57600., 0., 7571. / 16695., 393. / 640., -92097. / 339200., 187. / 2100., 1. / 40.}
//	return rk
//}
//
//// INTERNAL
//func (rk *RK) init(order int) {
//	rk.stages = order
//	rk.a = make([][]float32, order)
//	for i := range rk.a {
//		rk.a[i] = make([]float32, order)
//	}
//	rk.b = make([]float32, order)
//	rk.c = make([]float32, order)
//	rk.k = make([]nimble.Slice, order)
//
//	// initialize k:
//	for i := 0; i < rk.stages; i++ {
//		rk.k[i] = NewTensor(sim.Backend, sim.mDev.size)
//	}
//
//	rk.m0 = NewTensor(sim.Backend, sim.mDev.size)
//	rk.mbackup = NewTensor(sim.Backend, sim.mDev.size)
//}
//
//func (rk *RK) initAdaptive(errororder float64) {
//	rk.b2 = make([]float32, rk.stages)
//	rk.errororder = errororder
//}
//
//// INTERNAL
//func (rk *RK) free() {
//	for i := range rk.k {
//		rk.k[i].Free()
//		rk.k[i] = nil
//	}
//}
//
//// INTERNAL
//func newRK(order int) *RK {
//	rk := new(RK)
//	rk.init(order)
//	return rk
//}
//
//// On a bad time step (error too big),
//// do not re-try the step more than this number of times.
//const MAX_STEP_TRIALS = 10
//
//func (rk *RK) step() {
//
//	order := rk.stages
//	k := rk.k
//	time0 := rk.time
//	c := rk.c
//	m := rk.mDev
//	m1 := rk.m0
//	a := rk.a
//
//	//	// The thermal noise is assumed constant during the step.
//	//	if rk.input.temp != 0 {
//	//		rk.updateTempNoise(rk.dt)
//	//	}
//
//	TensorCopyOn(m, rk.mbackup)
//	goodstep := false
//	trials := 0
//
//	// Try to take a step with the current dt.
//	// If the step fails (error too big),
//	// then cut dt and try again (at most MAX_STEP_TRIALS times)
//	for !goodstep && trials < MAX_STEP_TRIALS {
//		trials++
//		for i := 0; i < order; i++ {
//
//			// Calculate torque
//			if rk.fsal && i == 0 && rk.fsal_initiated {
//				//FSAL: First step of this stage
//				//is the same as the last step of the previous stage.
//				TensorCopyOn(k[order-1], k[0])
//			} else {
//				rk.time = time0 + float64(c[i]*rk.dt)
//				TensorCopyOn(m, m1)
//				for j := 0; j < order; j++ {
//					if a[i][j] != 0. {
//						rk.MAdd(m1, rk.dt*a[i][j], k[j])
//					}
//				}
//				if rk.fsal {
//					// update energy at the end of the step for fsal solvers
//					if rk.wantEnergy && i == order-1 {
//						rk.calcHeffEnergy(m1, k[i])
//					} else {
//						rk.calcHeff(m1, k[i])
//					}
//				} else {
//					// update energy at the beginning of the step for non-fsal solvers
//					if rk.wantEnergy && i == 0 {
//						rk.calcHeffEnergy(m1, k[i])
//					} else {
//						rk.calcHeff(m1, k[i])
//					}
//				}
//
//				rk.Torque(m1, k[i])
//				rk.fsal_initiated = true
//			}
//
//			// After having calculated the first torque (k[0]),
//			// dt has not actually been used yet. This is the
//			// last chance to estimate whether the time step is too large
//			// and possibly reduce it.
//			// This cannot be done with finite temperature!
//			if rk.input.temp == 0 && i == 0 {
//				if rk.b2 == nil { // means no step control based on error estimate
//					rk.dt = rk.input.dt
//				}
//				assert(c[i] == 0)
//				maxTorque := rk.Reduce(k[0])
//				rk.Sim.torque = maxTorque // save centrally so it can be output
//				dm := rk.dt * maxTorque
//				// Do not make the magnetization step smaller than minDm
//				if dm < rk.input.minDm {
//					rk.dt = rk.input.minDm / maxTorque
//				}
//				// Do not make the time step smaller than minDt
//				if rk.dt*rk.UnitTime() < rk.input.minDt {
//					rk.dt = rk.input.minDt / rk.UnitTime()
//				}
//				// maxDm has priority over minDm (better safe than sorry)
//				if rk.input.maxDm != 0 && dm > rk.input.maxDm {
//					rk.dt = rk.input.maxDm / maxTorque
//				}
//				// maxDt has priority over minDt (better safe than sorry)
//				if rk.input.maxDt != 0. && rk.dt*rk.UnitTime() > rk.input.maxDt {
//					rk.dt = rk.input.maxDt / rk.UnitTime()
//				}
//				checkdt(rk.dt)
//			}
//		}
//
//		rk.time = time0 + float64(rk.dt) // THIS IS THE DT OF THIS LAST STEP
//		// IT WILL NOW BE UDPATED FOR THE NEXT STEP
//
//		//Lowest-order solution for error estimate, if applicable
//		//from now, m1 = lower-order solution
//		if rk.b2 != nil {
//			TensorCopyOn(m, m1)
//			for i := range k {
//				if rk.b2[i] != 0. {
//					rk.MAdd(m1, rk.b2[i]*rk.dt, k[i])
//				}
//			}
//		}
//
//		//Highest-order solution (m)
//		//TODO: not 100% efficient, too many adds
//		//	fmt.Println("m", m)
//		for i := range k {
//			//	fmt.Println("k ", i, " ", k[i])
//			if rk.b[i] != 0. {
//				rk.MAdd(m, rk.b[i]*rk.dt, k[i])
//			}
//		}
//
//		//calculate error and adapt step size if applicable
//		if rk.b2 != nil {
//			rk.MAdd(m1, -1, m) // make difference between high- and low-order solution TODO: should be difference between k's
//			error := rk.Reduce(m1)
//			rk.stepError = error
//
//			// calculate new step
//			assert(rk.input.maxError != 0.)
//			//TODO: what is the pre-factor of the error estimate?
//			factor := float32(math.Pow(float64(rk.input.maxError/error), 1./rk.errororder))
//
//			// do not increase by time step by more than 100%
//			if factor > 2. {
//				factor = 2.
//			}
//			// do not decrease to less than 1%
//			if factor < 0.01 {
//				factor = 0.01
//			}
//
//			rk.dt = rk.dt * factor
//
//			checkdt(rk.dt)
//			//undo bad steps
//			// if we want the energy to be calculated on-the-fly, we make sure it always drops
//			if error > 2*rk.input.maxError || (rk.wantEnergy && rk.energy > prevEnergy) {
//				TensorCopyOn(rk.mbackup, m)
//				goodstep = false
//				//fmt.Println("bad step")
//			} else {
//				goodstep = true
//				// Do not make the time step smaller than minDt
//				if rk.dt*rk.UnitTime() < rk.input.minDt {
//					rk.dt = rk.input.minDt / rk.UnitTime()
//				}
//				// maxDt has priority over minDt (better safe than sorry)
//				if rk.input.maxDt != 0. && rk.dt*rk.UnitTime() > rk.input.maxDt {
//					rk.dt = rk.input.maxDt / rk.UnitTime()
//				}
//			}
//		}
//	}
//
//	//rk.time = time0 // will be incremented by simrun.go
//	rk.Normalize(m)
//}
//
//// debug
//func checkdt(dt float32) {
//	if math.IsNaN(float64(dt)) {
//		panic("dt = NaN")
//	}
//}
//
//func (rk *RK) String() (str string) {
//	defer func() { recover(); return }()
//
//	for i := 0; i < rk.stages; i++ {
//		str += fmt.Sprint(rk.c[i]) + "\t|\t"
//		for j := 0; j < rk.stages; j++ {
//			str += fmt.Sprint(rk.a[i][j]) + "\t"
//		}
//		str += "\n"
//	}
//	str += "----\n\t|\t"
//	for i := 0; i < rk.stages; i++ {
//		str += fmt.Sprint(rk.b[i]) + "\t"
//	}
//	if rk.b2 != nil {
//		str += "\n"
//		for i := 0; i < rk.stages; i++ {
//			str += fmt.Sprint(rk.b2[i]) + "\t"
//		}
//	}
//	return str
//}

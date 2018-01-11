package engine

// Minimize follows the steepest descent method as per Exl et al., JAP 115, 17D118 (2014).

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	DmSamples int     = 10   // number of dm to keep for convergence check
	StopMaxDm float64 = 1e-6 // stop minimizer if sampled dm is smaller than this
)

func init() {
	DeclFunc("Minimize", Minimize, "Use steepest conjugate gradient method to minimize the total energy")
	DeclVar("MinimizerStop", &StopMaxDm, "Stopping max dM for Minimize")
	DeclVar("MinimizerSamples", &DmSamples, "Number of max dM to collect for Minimize convergence check.")
}

// fixed length FIFO. Items can be added but not removed
type fifoRing struct {
	count int
	tail  int // index to put next item. Will loop to 0 after exceeding length
	data  []float64
}

func FifoRing(length int) fifoRing {
	return fifoRing{data: make([]float64, length)}
}

func (r *fifoRing) Add(item float64) {
	r.data[r.tail] = item
	r.count++
	r.tail = (r.tail + 1) % len(r.data)
	if r.count > len(r.data) {
		r.count = len(r.data)
	}
}

func (r *fifoRing) Max() float64 {
	max := r.data[0]
	for i := 1; i < r.count; i++ {
		if r.data[i] > max {
			max = r.data[i]
		}
	}
	return max
}

type Minimizer struct {
	k      *data.Slice // torque saved to calculate time step
	lastDm fifoRing
	h      float32
}

func (mini *Minimizer) Step() {
	m := M.Buffer()
	size := m.Size()
	k := mini.k
	h := mini.h

	// save original magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	// make descent
	cuda.Minimize(m, m0, k, h)

	// calculate new torque for next step
	k0 := cuda.Buffer(3, size)
	defer cuda.Recycle(k0)
	data.Copy(k0, k)
	torqueFn(k)
	setMaxTorque(k) // report to user

	// just to make the following readable
	dm := m0
	dk := k0

	// calculate step difference of m and k
	cuda.Madd2(dm, m, m0, 1., -1.)
	cuda.Madd2(dk, k, k0, -1., 1.) // reversed due to LLNoPrecess sign

	// get maxdiff and add to list
	max_dm := cuda.MaxVecNorm(dm)
	mini.lastDm.Add(max_dm)
	setLastErr(mini.lastDm.Max()) // report maxDm to user as LastErr

	// adjust next time step
	var nom, div float32
	if NSteps%2 == 0 {
		nom = cuda.Dot(dm, dm)
		div = cuda.Dot(dm, dk)
	} else {
		nom = cuda.Dot(dm, dk)
		div = cuda.Dot(dk, dk)
	}
	if div != 0. {
		mini.h = nom / div
	} else { // in case of division by zero
		mini.h = 1e-4
	}

	M.normalize()

	// as a convention, time does not advance during relax
	NSteps++
}

func (mini *Minimizer) Free() {
	cuda.Recycle(mini.k)
}

func Minimize() {
	Refer("exl2014")
	SanityCheck()
	// Save the settings we are changing...
	prevType := solvertype
	prevFixDt := FixDt
	prevPrecess := Precess
	t0 := Time

	relaxing = true // disable temperature noise

	// ...to restore them later
	defer func() {
		SetSolver(prevType)
		FixDt = prevFixDt
		Precess = prevPrecess
		Time = t0

		relaxing = false
	}()

	Precess = false // disable precession for torque calculation
	// remove previous stepper
	if stepper != nil {
		stepper.Free()
	}

	// set stepper to the minimizer
	size := M.Buffer().Size()
	mini := Minimizer{
		h:      1e-4,
		k:      cuda.Buffer(3, size),
		lastDm: FifoRing(DmSamples)}
	stepper = &mini

	// calculate initial torque
	torqueFn(mini.k)

	cond := func() bool {
		return (mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm)
	}

	RunWhile(cond)
	pause = true
}

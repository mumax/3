package engine

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
)

// Adaptive Heun solver.
type Heun struct{}

// Adaptive Heun method, can be used as solver.Step
func (*Heun) Step() {
	y := M.Buffer()
	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)

	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si * GammaLL)
	util.Assert(dt > 0)

	// stage 1
	torqueFn(dy0)
	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy

	// stage 2
	dy := cuda.Buffer(3, y.Size())
	defer cuda.Recycle(dy)
	Time += Dt_si
	torqueFn(dy)

	err := cuda.MaxVecDiff(dy0, dy) * float64(dt)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
		M.normalize()
		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./2.))
		setLastErr(err)
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time -= Dt_si
		cuda.Madd2(y, y, dy0, 1, -dt)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./3.))
	}
}

func (*Heun) Free() {}

// StepCaptureBody performs the GPU-side work of one fixed-step Heun step
// (FixDt != 0), suitable for recording into a CUDA Graph (see RunGraph).
//
// Compared to Step, it omits:
//   - Time/NSteps bookkeeping, which RunGraph's replay loop does itself
//     once per graph launch instead;
//   - err := cuda.MaxVecDiff(...), adaptDt and setLastErr, which are
//     non-load-bearing when FixDt != 0 (the accept branch is always taken,
//     adaptDt is a no-op, and setLastErr only affects the reported LastErr).
//
// The MaxVecDiff omission is not just an optimization: cuda.MaxVecDiff reads
// its result back to the host synchronously, which is unsupported while a
// stream is being captured and would abort the capture outright.
func (*Heun) StepCaptureBody() {
	util.Assert(FixDt != 0)
	y := M.Buffer()
	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)

	dt := float32(FixDt * GammaLL)
	util.Assert(dt > 0)

	// stage 1
	torqueFn(dy0)
	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy

	// stage 2
	dy := cuda.Buffer(3, y.Size())
	defer cuda.Recycle(dy)
	torqueFn(dy)

	cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
	M.normalize()
}

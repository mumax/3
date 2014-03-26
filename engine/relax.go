package engine

import (
	"github.com/mumax/3/cuda"
	"math"
)

func init() {
	DeclFunc("Relax", Relax, "Try to minimize the total energy")
}

func Relax() {
	SanityCheck()
	pause = false

	prevType := solvertype
	prevErr := MaxErr
	prevFixDt := FixDt
	prevPrecess := Precess

	SetSolver(BOGAKISHAMPINE)
	FixDt = 0
	Precess = false
	defer func() {
		SetSolver(prevType)
		MaxErr = prevErr
		FixDt = prevFixDt
		Precess = prevPrecess
	}()

	solver := stepper.(*RK23)
	avgTorque := func() float32 {
		return cuda.Dot(solver.k1, solver.k1)
	}

	const N = 3
	relaxSteps(N)
	E0 := GetTotalEnergy()
	relaxSteps(N)
	E1 := GetTotalEnergy()
	for E1 < E0 && !pause {
		relaxSteps(N)
		E0, E1 = E1, GetTotalEnergy()
	}

	var T0, T1 float32 = 0, avgTorque()

	for MaxErr > 1e-9 && !pause {
		MaxErr /= math.Sqrt2
		relaxSteps(N)
		T0, T1 = T1, avgTorque()
		for T1 < T0 && !pause {
			relaxSteps(1)
			T0, T1 = T1, avgTorque()
		}
	}

	pause = true
}

// take n steps without setting pause when done or advancing time
func relaxSteps(n int) {
	stop := NSteps + n
	t0 := Time
	for NSteps < stop && !pause {
		select {
		default:
			step()
			Time = t0
		// accept tasks form Inject channel
		case f := <-Inject:
			f()
		}
	}
}

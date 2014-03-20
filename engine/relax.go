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
	rSteps(N)
	E0 := GetTotalEnergy()
	rSteps(N)
	E1 := GetTotalEnergy()
	for E1 < E0 && !pause {
		rSteps(N)
		E0, E1 = E1, GetTotalEnergy()
	}

	var T0, T1 float32 = 0, avgTorque()

	for MaxErr > 1e-9 && !pause {
		MaxErr /= math.Sqrt2
		//util.Println(MaxErr)
		//util.Println(avgTorque())
		rSteps(N)
		T0, T1 = T1, avgTorque()
		for T1 < T0 && !pause {
			rSteps(1)
			T0, T1 = T1, avgTorque()
		}
	}

	pause = true
}

// take n steps without setting pause when done.
func rSteps(n int) {
	stop := NSteps + n
	for NSteps < stop && !pause {
		select {
		default:
			step()
		// accept tasks form Inject channel
		case f := <-Inject:
			f()
		}
	}
}

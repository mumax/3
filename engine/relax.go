package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
	"math"
)

func init() {
	DeclFunc("Relax", Relax, "Try to minimize the total energy")
}

func Relax() {
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
	//	maxTorque := func() float64 {
	//		return cuda.MaxVecNorm(solver.k1)
	//	}
	avgTorque := func() float64 {
		return math.Sqrt(float64(cuda.Dot(solver.k1, solver.k1)))
	}

	const N = 3
	Steps(N)
	E0 := GetTotalEnergy()
	Steps(N)
	E1 := GetTotalEnergy()
	for E1 < E0 {
		Steps(N)
		E0, E1 = E1, GetTotalEnergy()
	}

	T0 := 0.
	T1 := avgTorque()

	for MaxErr > 1e-9 {
		MaxErr /= math.Sqrt2
		util.Println(MaxErr)
		Steps(N)
		T0, T1 = T1, avgTorque()
		for T1 < T0 {
			Steps(1)
			T0, T1 = T1, avgTorque()
		}
	}

}

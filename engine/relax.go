package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
	"math"
)

func init() {
	DeclFunc("Relax", Relax, "Try to minimize the total energy until maxTorque limit")
}

func Relax() {
	prevType := solvertype
	prevErr := MaxErr
	prevFixDt := FixDt

	SetSolver(BOGAKISHAMPINE)
	FixDt = 0
	defer func() {
		SetSolver(prevType)
		MaxErr = prevErr
		FixDt = prevFixDt
	}()

	solver := stepper.(*RK23)
	maxTorque := func() float64 {
		return cuda.MaxVecNorm(solver.k1)
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
	T1 := maxTorque()

	for MaxErr > 1e-9 {
		MaxErr /= math.Sqrt2
		util.Println(MaxErr)
		Steps(N)
		T0, T1 = T1, maxTorque()
		for T1 < T0 {
			Steps(1)
			T0, T1 = T1, maxTorque()
		}
	}

}

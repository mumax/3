package engine

// Relax tries to find the minimum energy state.

import (
	"github.com/mumax/3/cuda"
	"math"
)

var thresholdTorque := 1e-3; //to be replaced by a sane default value

func init() {
	DeclFunc("Relax", Relax, "Try to minimize the total energy")
	DeclVar("RelaxThresholdTorque", &thresholdTorque, "Stopping maximum torque threshold for relax()")

}

// are we relaxing?
var relaxing = false

func Relax() {
	SanityCheck()
	pause = false

	// Save the settings we are changing...
	prevType := solvertype
	prevErr := MaxErr
	prevFixDt := FixDt
	prevPrecess := Precess

	// ...to restore them later
	defer func() {
		SetSolver(prevType)
		MaxErr = prevErr
		FixDt = prevFixDt
		Precess = prevPrecess
		relaxing = false
		//	Temp.upd_reg = prevTemp
		//	Temp.invalidate()
		//	Temp.update()
	}()

	// Set good solver for relax
	SetSolver(BOGAKISHAMPINE)
	FixDt = 0
	Precess = false
	relaxing = true

	// Minimize energy: take steps as long as energy goes down.
	// This stops when energy reaches the numerical noise floor.
	const N = 3 // evaluate energy (expensive) every N steps
	relaxSteps(N)
	E0 := GetTotalEnergy()
	relaxSteps(N)
	E1 := GetTotalEnergy()
	for E1 < E0 && !pause {
		relaxSteps(N)
		E0, E1 = E1, GetTotalEnergy()
	}

	// Now we are already close to equilibrium, but energy is too noisy to be used any further.
	// So now we minimize the maximum torque which is less noisy.
	solver := stepper.(*RK23)
	defer stepper.Free() // purge previous rk.k1 because FSAL will be dead wrong.
	
	maxTorque := func() float32 {
		return cuda.MaxVecNorm(solver.k1) //perhaps replace by (faster?) sqrt(3)*cuda.MaxAbs(...)? 
    	}
	// run as long as the max torque is above threshold. Then increase the accuracy and step more.
	for !pause {
		for maxTorque() < thresholdTorque && !pause {
		    relaxSteps(N)
		}
		MaxErr /= math.Sqrt2
		if MaxErr < 1e-9 { 
			break
		}
    	}
	pause = true
}

// take n steps without setting pause when done or advancing time
func relaxSteps(n int) {
	t0 := Time
	stop := NSteps + n
	cond := func() bool { return NSteps < stop }
	const output = false
	runWhile(cond, output)
	Time = t0
}

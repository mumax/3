package engine

// Relax tries to find the minimum energy state.

import (
	"math"
	"time"

	"github.com/mumax/3/cuda"
)

var (
	// Stopping relax Maxtorque in T. The user can check MaxTorque for sane values (e.g. 1e-3).
	// If set to <=0, relax() will stop when the average torque is steady or increasing.
	RelaxTorqueThreshold float64 = -1.
	RelaxWallClockTime   float64 = -1.0 // wall-clock time limit for Relax
	RelaxConverged       bool           // true if Relax converged, and false if the maximum wall-clock time is reached
)

func init() {
	DeclFunc("Relax", Relax, "Try to minimize the total energy. Returns true if convergence is reached, or false if the wall-clock time limit is exceeded. The wall-clock time limit is disabled by default.")
	DeclVar("RelaxTorqueThreshold", &RelaxTorqueThreshold, "MaxTorque threshold for relax(). If set to -1 (default), relax() will stop when the average torque is steady or increasing.")
	DeclVar("RelaxWallClockTime", &RelaxWallClockTime, "Wall-clock time limit (seconds) for Relax that will interrupt the relaxation if exceeded. Set to -1 (default) to disable.")
}

// are we relaxing?
var relaxing = false

func Relax() bool {

	// if wall-clock time is zero, skip Relaxing entirely (zero steps), and don't change any settings
	RelaxConverged = false
	TimerStart := time.Now()
	if RelaxWallClockTime == 0 {
		RelaxConverged = false
		return RelaxConverged
	}

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
	SetSolver(BOGACKISHAMPINE)
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
	for (E1 < E0 && !pause) && WallclockTimer(TimerStart, RelaxWallClockTime) {
		relaxSteps(N)
		E0, E1 = E1, GetTotalEnergy()
	}

	// Now we are already close to equilibrium, but energy is too noisy to be used any further.
	// So now we minimize the torque which is less noisy.
	solver := stepper.(*RK23)
	defer stepper.Free() // purge previous rk.k1 because FSAL will be dead wrong.

	maxTorque := func() float64 {
		return cuda.MaxVecNorm(solver.k1)
	}
	avgTorque := func() float32 {
		return cuda.Dot(solver.k1, solver.k1)
	}

	if RelaxTorqueThreshold > 0 {
		// run as long as the max torque is above threshold. Then increase the accuracy and step more.
		for !pause {
			for (maxTorque() > RelaxTorqueThreshold && !pause) && WallclockTimer(TimerStart, RelaxWallClockTime) {
				relaxSteps(N)
			}
			MaxErr /= math.Sqrt2
			if MaxErr < 1e-9 {
				break
			}
		}
	} else {
		// previous (<jan2018) behaviour: run as long as torque goes down. Then increase the accuracy and step more.
		// if MaxErr < 1e-9, this code won't run.
		var T0, T1 float32 = 0, avgTorque()
		// Step as long as torque goes down. Then increase the accuracy and step more.
		for (MaxErr > 1e-9 && !pause) && WallclockTimer(TimerStart, RelaxWallClockTime) {
			MaxErr /= math.Sqrt2
			relaxSteps(N) // TODO: Play with other values
			T0, T1 = T1, avgTorque()
			for T1 < T0 && !pause {
				relaxSteps(N) // TODO: Play with other values
				T0, T1 = T1, avgTorque()
			}
		}
	}

	pause = true
	RelaxConverged = (RelaxTorqueThreshold > 0 && (maxTorque() <= RelaxTorqueThreshold || MaxErr < 1e-9)) || (RelaxTorqueThreshold <= 0 && MaxErr <= 1e-9)
	stepper.Free()
	return RelaxConverged
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

package engine

import (
	"math"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Expose to user scripts
func init() {
	DeclFunc("MinimizeForSeconds", MinimizeForSeconds, "Minimize for a fixed wall-clock time (int seconds)")
	DeclFunc("RelaxForSeconds", RelaxForSeconds, "Relax for a fixed wall-clock time (int seconds)")
	DeclFunc("RunForSeconds", RunForSeconds, "Run the simulation for a fixed wall-clock time (int seconds)")
	DeclFunc("CurrentMag", CurrentMag, "Returns the current magnetization as a Config. E.g. CurrentMag().Add(0.1, RandomMagSeed(123)) will return a Config with the current magnetization plus some noise.")
	DeclFunc("RunSequence", RunSequence, " (total wallclock time, noise, wallclock1, wallclock 2, wallclock3, seed). Runs a sequence of steps: 1) Add noise to current magnetization as CurrentMag().Add(noise, RandomMagSeed(seed)), 2) Run for wallclock1 seconds, 3) Relax for wallclock2 seconds, 4) Minimize for wallclock3 seconds. The sequence is repeated until convergence or total wall-clock time limit is reached.")
}

func MinimizeForSeconds(seconds int) (converged bool) {

	start := time.Now()
	converged = false
	if seconds == 0 {
		converged = false
		return converged
	}

	Refer("exl2014")
	SanityCheck()

	// Save solver state
	prevType := solvertype
	prevFixDt := FixDt
	prevPrecess := Precess
	t0 := Time

	relaxing = true

	defer func() {
		SetSolver(prevType)
		FixDt = prevFixDt
		Precess = prevPrecess
		Time = t0
		relaxing = false
	}()

	Precess = false

	if stepper != nil {
		stepper.Free()
	}

	mini := Minimizer{
		h:      1e-4,
		k:      nil,
		lastDm: FifoRing(DmSamples)}
	stepper = &mini

	cond := func() bool {
		return (mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm) && (time.Since(start) < time.Duration(seconds)*time.Second)
	}

	RunWhile(cond)
	pause = true

	converged = time.Since(start) <= time.Duration(seconds)*time.Second
	stepper.Free()
	return converged
}

func RelaxForSeconds(seconds int) (converged bool) {

	converged = false
	start := time.Now()
	if seconds == 0 {
		converged = false
		return converged
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
	for E1 < E0 && !pause {
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
			for maxTorque() > RelaxTorqueThreshold && !pause && time.Since(start) < time.Duration(seconds)*time.Second {
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
		for MaxErr > 1e-9 && !pause && time.Since(start) < time.Duration(seconds)*time.Second {
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
	converged = time.Since(start) <= time.Duration(seconds)*time.Second
	stepper.Free()
	return converged
}

func RunForSeconds(seconds int) {
	deadline := time.Now().Add(time.Duration(seconds) * time.Second)

	RunWhile(func() bool {
		return time.Now().Before(deadline)
	})
}

func CurrentMag() Config {

	d := Mesh().CellSize()
	size := Mesh().Size()

	Nx, Ny, Nz := size[X], size[Y], size[Z]

	Lx := float64(Nx) * d[X]
	Ly := float64(Ny) * d[Y]
	Lz := float64(Nz) * d[Z]

	mSlice := (&M).Buffer().HostCopy()

	return func(x, y, z float64) data.Vector {

		ix := int(math.Floor((x + 0.5*Lx) / d[X]))
		iy := int(math.Floor((y + 0.5*Ly) / d[Y]))
		iz := int(math.Floor((z + 0.5*Lz) / d[Z]))

		if ix < 0 {
			ix = 0
		}
		if ix >= Nx {
			ix = Nx - 1
		}
		if iy < 0 {
			iy = 0
		}
		if iy >= Ny {
			iy = Ny - 1
		}
		if iz < 0 {
			iz = 0
		}
		if iz >= Nz {
			iz = Nz - 1
		}

		return data.Vector{
			mSlice.Get(X, ix, iy, iz),
			mSlice.Get(Y, ix, iy, iz),
			mSlice.Get(Z, ix, iy, iz),
		}
	}

}

func RunSequence(totalWalltimeSec int, noise float64, wallclock1, wallclock2, wallclock3, seed int) bool {

	if wallclock1 == 0 && wallclock2 == 0 && wallclock3 != 0 {
		util.Log("warning: no run or relax called before minimize")
	}

	// Apply noise to current magnetization
	if noise > 0 {
		(&M).Set(CurrentMag().Add(noise, RandomMagSeed(seed)))
	}

	start := time.Now()
	totalWalltime := time.Duration(totalWalltimeSec) * time.Second

	converged := false

	for !converged || time.Since(start) < totalWalltime {

		RunForSeconds(wallclock1)

		if time.Since(start) >= totalWalltime {
			break
		}

		b := RelaxForSeconds(wallclock2)

		if b {
			converged = true
			break
		}

		a := MinimizeForSeconds(wallclock3)

		if a {
			converged = true
			break
		}
		if time.Since(start) >= totalWalltime {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	return converged
}

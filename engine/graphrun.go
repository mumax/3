package engine

// CUDA Graph capture, based on paper C.-Y. You, J. Magn. 31, 204-213 (2026).
//
// nsys profiling showed that for small/medium grids, 75~79% of step time is
// spent in cuLaunchKernel driver calls (~27 per step), not in the kernels
// themselves. StepsGraph captures the torque-evaluation kernel sequences as
// CUDA Graphs and replays them with a single cuGraphLaunch each, instead of
// ~13 individual cuLaunchKernel calls per evaluation, resulting in a 3-5x
// speedup on small grids (depending on the solver used).

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var EnableCUDAgraphs = true

func init() {
	DeclFunc("StepsGraph", StepsGraph, "Like Steps, but captures the torque-evaluation kernels as CUDA Graphs and "+
		"replays them for the remaining steps. Supports Heun, RK23, RK45DP (default), RK56 and BackwardEuler.")
	DeclVar("EnableCUDAgraphs", &EnableCUDAgraphs, "Enables CUDA Graphs, greatly improving performance of Run() and Steps() on small grids (default=true)<br>NOTE: graphs are only used if Temp=0, NoDemagSpins=0 and no custom/time-varying fields are defined.")
}

// StepsGraph performs n further steps, capturing the GPU work of the torque
// evaluation(s) into one or more CUDA Graphs and replaying them for the
// remaining steps. Heun (solver(2)), RK23 (solver(3)), RK45DP (solver(5), the
// default), RK56 (solver(6)) and BackwardEuler (solver(-1)) are supported.
//
// In all solvers (except Heun with FixDt != 0), dt and/or the accept/reject
// decision change every step and depend on a blocking host readback (e.g.,
// MaxVecDiff/MaxVecNorm) that stream capture cannot record (see Sec. 4.1 of
// You2026). In these cases, only the dt-independent torque evaluations are
// captured as separate small graphs ("split graph"), while the dt-dependent
// Madd*/error-estimate/normalize/adaptDt/accept-reject/FSAL steps run as
// ordinary (uncaptured) calls in the replay loop, exactly as in the
// corresponding Step().
func StepsGraph(n int) {
	if n <= 0 {
		return
	}

	assertGraphCompatible()

	stop := NSteps + n
	condition := func() bool { return NSteps < stop }

	var fellBack bool
	switch s := stepper.(type) {
	case *Heun:
		if FixDt != 0 {
			fellBack = runGraphHeunFixedDt(s, condition)
		} else {
			fellBack = runGraphHeunAdaptive(s, condition)
		}
	case *RK45DP:
		fellBack = runGraphRK45DP(s, condition)
	case *RK23:
		fellBack = runGraphRK23(s, condition)
	case *RK56:
		fellBack = runGraphRK56(s, condition)
	case *BackwardEuler:
		fellBack = runGraphBackwardEuler(s, condition)
	default:
		util.AssertMsg(false, "StepsGraph: requires Heun (solver(2)), RK23 (solver(3)), RK45DP (solver(5), the default), RK56 (solver(6)), or BackwardEuler (solver(-1))")
	}
	// An Inject (GUI/script) call mid-replay may have changed state the
	// captured graphs depend on; see checkInject. Finish the remaining steps
	// the safe (non-graph) way.
	if fellBack {
		RunWhile(condition)
	}
}

// Attempt to run the simulation using the CUDA Graph capture/replay path
// instead of RunWhile, returning true if it did so. It returns false (without
// side effects) if the current configuration is not graphCompatible(), the
// mesh is too large to benefit (graphWorthwhile()), or the current solver has
// no graph runner. If false is returned, the caller (Run/Steps) should fall
// back to RunWhile(condition).
func tryRunGraph(condition func() bool) bool {
	if !graphCompatible() || !graphWorthwhile() || !EnableCUDAgraphs {
		return false
	}

	var fellBack bool
	switch s := stepper.(type) {
	case *Heun:
		if FixDt != 0 {
			fellBack = runGraphHeunFixedDt(s, condition)
		} else {
			fellBack = runGraphHeunAdaptive(s, condition)
		}
	case *RK45DP:
		fellBack = runGraphRK45DP(s, condition)
	case *RK23:
		fellBack = runGraphRK23(s, condition)
	case *RK56:
		fellBack = runGraphRK56(s, condition)
	case *BackwardEuler:
		fellBack = runGraphBackwardEuler(s, condition)
	default:
		return false
	}
	// An Inject (GUI/script) call mid-replay may have changed state the
	// captured graphs depend on; see checkInject. Finish the remaining steps
	// the safe (non-graph) way.
	if fellBack {
		RunWhile(condition)
	}
	return true
}

// Panics if the current configuration is outside the envelope StepsGraph has
// been verified for: constant excitation, Temp == 0, no custom field terms,
// and no region-wise time-dependent material parameters (see Sec. 6.3 of
// You2026). Outside this envelope, a captured graph would replay a stale
// kernel sequence (fixed Time/RNG state/LUTs/etc.) every step, silently
// producing wrong results instead of erroring out.
func assertGraphCompatible() {
	if reason := graphIncompatibilityReason(); reason != "" {
		util.AssertMsg(false, "StepsGraph: "+reason)
	}
}

// Returns true if the current configuration is within the envelope
// assertGraphCompatible requires, without panicking. Used by the transparent
// Run()/Steps() graph dispatch (tryRunGraph in run.go) to decide,
// side-effect-free, whether the graph path may be used.
func graphCompatible() bool {
	return graphIncompatibilityReason() == ""
}

// Returns a human-readable reason why StepsGraph (or the transparent
// Run()/Steps() graph path) cannot be used with the current configuration,
// or the empty string "" if it can.
func graphIncompatibilityReason() string {
	if !Temp.isZero() {
		return "Temp != 0 is not supported (thermal field)"
	}
	if !isTimeIndependent(B_ext) {
		return "B_ext must be time-independent (no extraTerms or time-dependent per-region value)"
	}
	if len(customTerms) != 0 {
		return "custom field terms (AddFieldTerm) are not supported"
	}
	if !NoDemagSpins.isZero() {
		// SetDemagField takes the setMaskedDemagField path (engine/demag.go),
		// which calls data.Copy on the geometry mask. data.Copy goes through
		// cuda.MemCpy, which calls cuda.Sync() (a blocking cuStreamSynchronize)
		// -- not permitted during stream capture (CUresult 900,
		// CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED), same class of issue as the
		// blocking MemCpyDtoH in MaxVecDiff/MaxVecNorm (see Box 1.3 of You2026).
		return "NoDemagSpins != 0 is not supported (masked demag field)"
	}

	// Material parameters feeding the captured effective-field
	// (AddExchangeField/AddAnisotropyField/AddMagnetoelasticField/
	// SetDemagField) and torque (torqueFn) kernels. If any of these is set
	// to a per-region time-dependent function (SetRegion/SetRegionFn with
	// an expression containing t), the LUT/scalar kernel argument baked
	// into the captured graph would be frozen at capture-time Time and
	// replayed unchanged forever.
	materialParams := []*regionwise{
		&Msat.regionwise, &Aex.regionwise, &Dind.regionwise, &Dbulk.regionwise,
		&Ku1.regionwise, &Ku2.regionwise, &Kc1.regionwise, &Kc2.regionwise, &Kc3.regionwise,
		&AnisU.regionwise, &AnisC1.regionwise, &AnisC2.regionwise,
		&B1.regionwise, &B2.regionwise,
		&Alpha.regionwise, &Xi.regionwise, &Pol.regionwise, &Lambda.regionwise,
		&EpsilonPrime.regionwise, &FrozenSpins.regionwise, &FreeLayerThickness.regionwise,
	}
	for _, p := range materialParams {
		if hasTimeDependentRegion(p) {
			return p.name + " must be time-independent (SetRegion/SetRegionFn with a function of t is not supported)"
		}
	}
	return ""
}

// graphMaxCells is the cell-count threshold above which the transparent
// Run()/Steps() graph path is not used, even if graphCompatible(). This value
// is chosen for CUDA Graph Capture to yield a >5% performance increase, based
// on the StepsGraph speedups reported in Fig. 3 of You2026:
//   - 512x512x1 (262,144 cells): ~1.3-1.5x
//   - 700x700x1 (490,000 cells): ~1.14x
//   - 800x800x1 (640,000 cells): ~1.06x
//   - 1024x1024x1 (1,048,576 cells): ~1.0-1.04x (no longer worthwhile).
const graphMaxCells = 800000

// Returns true if the current mesh is small enough for CUDA Graph capture to
// result in a meaningful speedup (see graphMaxCells).
func graphWorthwhile() bool {
	size := Mesh().Size()
	return size[0]*size[1]*size[2] <= graphMaxCells
}

// graphMinSteps is the minimum number of steps a Steps(n)/Run(seconds) call
// must cover for the transparent graph path to be attempted. Capturing and
// instantiating the split-graph(s) has a fixed one-time cost (stream capture
// + cuGraphInstantiate per graph); this heuristic ensures it is amortized
// over enough replayed steps to be worthwhile. Tunable; not load-bearing for
// correctness (StepsGraph itself has no such threshold).
const graphMinSteps = 20

// Returns true if p has been set to a time-dependent function in any region.
func hasTimeDependentRegion(p *regionwise) bool {
	for r := 0; r < NREGION; r++ {
		if p.upd_reg[r] != nil {
			return true
		}
	}
	return false
}

// Returns true if the excitation e is constant in time in all regions.
func isTimeIndependent(e *Excitation) bool {
	if len(e.extraTerms) > 0 {
		return false
	}
	for r := 0; r < NREGION; r++ {
		if e.perRegion.upd_reg[r] != nil {
			return false
		}
	}
	return true
}

// Perform one full torque evaluation on the default stream. This must be run
// once before any CUDA Graph capture begins, but after all cuda.Buffers that
// will be held for the lifetime of the capture/replay loop (e.g., m0, Err,
// k2..k8, dy0, dy...) have already been allocated.
//
// The reason for this is twofold.
// Firstly, cuMemAlloc is not allowed during CUDA stream capture (throws error
// 0xc0000005 Access Violation), but is used in the first call of SetTorque to
// lazily upload region-wise material parameter LUTs, compute/upload the demag
// kernel, and grow the cuda.Buffer pool to whatever scratch sizes the active
// field terms need.
// Secondly, as explained in Box 1.2 of You2026, cuMemAlloc only grows the
// cuda.Buffer pool to the high-water mark of buffers checked out so far.
// Hence, running warmupTorque before the solver buffers have been checked out
// would not allocate enough memory to store those solver buffers in addition
// to all buffers used in torqueFn(). Consequently, the internal temporaries
// would trigger a second cuMemAlloc, this time in-capture, causing a crash.
func warmupTorque(m *data.Slice) {
	scratch := cuda.Buffer(VECTOR, m.Size())
	defer cuda.Recycle(scratch)
	SetTorque(scratch)
}

// Non-blockingly check Inject for a pending GUI/script function. If this is
// the case, checkInject() runs the injected function and returns true.
//
// Since the injected function f may change the global state that the already-
// captured CUDA graphs depend on, in ways that graphIncompatibilityReason()
// may not even cover (e.g., resizing mesh, changing solver...), continuing
// to replay the graph may silently produce wrong results or cause a crash.
// Therefore, all runGraph* replay loops should call checkInject() once per
// iteration and stop replaying captured graphs immediately at any injection.
//
// Note that f may affect the solver global variable "pause". If pause==true,
// the runGraph* replay loop should exit normally (just like runWhile would).
// If pause==false, runGraph* should return true after interrupting its loop,
// such that its caller (StepsGraph/tryRunGraph) then finishes the remaining
// steps via RunWhile, which re-derives everything from the (possibly changed)
// global state and is correct regardless of what f changed.
func checkInject() bool {
	select {
	case f := <-Inject:
		f()
		return true
	default:
		return false
	}
}

// Implements StepsGraph/tryRunGraph for Heun (solver 2) FixDt != 0.
// Since every kernel's scalar arguments (dt, region LUTs, ...) are constant
// from step to step, the entire step can be captured once into a single graph
// and replayed unchanged for n steps.
func runGraphHeunFixedDt(heun *Heun, condition func() bool) bool {
	SanityCheck()
	pause = false

	util.Assert(FixDt != 0)
	dt := float32(FixDt * GammaLL)
	util.Assert(dt > 0)

	y := M.Buffer()

	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)
	dy := cuda.Buffer(3, y.Size())
	defer cuda.Recycle(dy)

	warmupTorque(M.Buffer())

	captureStream := cuda.EnterCaptureMode()
	demagConv().SetStream(captureStream)
	defer func() {
		demagConv().SetStream(cu.Stream(0))
		cuda.ExitCaptureMode(captureStream)
	}()

	cu.StreamBeginCapture(captureStream, cu.STREAM_CAPTURE_MODE_THREAD_LOCAL)
	torqueFn(dy0)                // stage 1
	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy
	torqueFn(dy)                 // stage 2
	cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
	M.normalize()
	graph := cu.StreamEndCapture(captureStream)
	defer graph.Destroy()

	exec := graph.Instantiate()
	defer exec.Destroy()

	fellBack := false
	for condition() && !pause {
		if checkInject() {
			if !pause {
				fellBack = true
			}
			break
		}

		exec.Launch(captureStream)
		captureStream.Synchronize()

		Time += FixDt
		NSteps++
		NEvals += 2 // each graph replay performs 2 torque evaluations
		for _, f := range postStep {
			f()
		}
		DoOutput()
	}
	pause = true
	return fellBack
}

// Implements StepsGraph/tryRunGraph for Heun (solver 2) for FixDt == 0,
// using the "split graph" approach (see Sec. 4.1 in You2026).
//
// Since the kernel sequence computing the torque is independent of dt and M
// (only the addresses of the dy0, dy and M buffers matter, which are fixed),
// it suffices to capture torqueFn(dy0) and torqueFn(dy) once before the loop,
// as small graphs execT0/execT1. The loop body then mirrors Heun.Step exactly
// but with torqueFn calls replaced by GraphExec.Launch calls. Other functions
// that can not be captured by graphs (e.g., Madd2, Madd3, MaxVecDiff...)
// remain interspersed between graph launches.
func runGraphHeunAdaptive(heun *Heun, condition func() bool) bool {
	SanityCheck()
	pause = false

	y := M.Buffer()

	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)
	dy := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy)

	warmupTorque(y)

	captureStream := cuda.EnterCaptureMode()
	demagConv().SetStream(captureStream)
	defer func() {
		demagConv().SetStream(cu.Stream(0))
		cuda.ExitCaptureMode(captureStream)
	}()

	cu.StreamBeginCapture(captureStream, cu.STREAM_CAPTURE_MODE_THREAD_LOCAL)
	torqueFn(dy0)
	graphT0 := cu.StreamEndCapture(captureStream)
	defer graphT0.Destroy()
	execT0 := graphT0.Instantiate()
	defer execT0.Destroy()

	cu.StreamBeginCapture(captureStream, cu.STREAM_CAPTURE_MODE_THREAD_LOCAL)
	torqueFn(dy)
	graphT1 := cu.StreamEndCapture(captureStream)
	defer graphT1.Destroy()
	execT1 := graphT1.Instantiate()
	defer execT1.Destroy()

	fellBack := false
	for condition() && !pause {
		if checkInject() {
			if !pause {
				fellBack = true
			}
			break
		}

		dt := float32(Dt_si * GammaLL)
		util.Assert(dt > 0)

		// stage 1
		execT0.Launch(captureStream)
		NEvals++
		cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy0

		// stage 2
		Time += Dt_si
		execT1.Launch(captureStream)
		NEvals++

		err := cuda.MaxVecDiff(dy0, dy) * float64(dt)

		// adjust next time step
		if err < MaxErr || Dt_si <= MinDt { // mindt check to avoid infinite loop
			// step OK
			cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
			M.normalize()
			NSteps++
			adaptDt(math.Pow(MaxErr/err, 1./2.))
			setLastErr(err)
		} else {
			// undo bad step
			Time -= Dt_si
			cuda.Madd2(y, y, dy0, 1, -dt)
			NUndone++
			adaptDt(math.Pow(MaxErr/err, 1./3.))
		}

		for _, f := range postStep {
			f()
		}
		DoOutput()
	}
	pause = true
	return fellBack
}

// Implements StepsGraph/tryRunGraph for Dormand-Prince (solver 5),
// using the "split graph" approach (see Sec. 4.1 in You2026).
//
// Since the kernel sequence computing the torque is independent of dt and M
// (only the addresses of the buffers matter, which are fixed), it suffices to
// capture torqueFn(k2)..torqueFn(k6) once before the loop, as small graphs
// execK2..execK6. The loop body then mirrors RK45DP.Step exactly but with
// torqueFn calls replaced by GraphExec.Launch calls. Other functions that can
// not be captured by graphs (e.g., Madd2, MaxVecDiff...) remain interspersed
// between graph launches.
func runGraphRK45DP(rk *RK45DP, condition func() bool) bool {
	SanityCheck()
	pause = false

	rk.Free() // mirror RunWhile's stepper.Free(): start from a clean state

	m := M.Buffer()
	size := m.Size()

	// upon resize: remove wrongly sized k1
	if rk.k1.Size() != m.Size() {
		rk.Free()
	}
	// first step ever: one-time k1 init and eval
	if rk.k1 == nil {
		rk.k1 = cuda.NewSlice(3, size)
		torqueFn(rk.k1)
	}

	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	Err := cuda.Buffer(3, size)
	defer cuda.Recycle(Err)

	k2, k3, k4, k5, k6 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)
	defer cuda.Recycle(k5)
	defer cuda.Recycle(k6)
	k7 := k2 // re-use k2, as in RK45DP.Step()

	warmupTorque(m)

	captureStream := cuda.EnterCaptureMode()
	demagConv().SetStream(captureStream)
	defer func() {
		demagConv().SetStream(cu.Stream(0))
		cuda.ExitCaptureMode(captureStream)
	}()

	dsts := [5]*data.Slice{k2, k3, k4, k5, k6}
	var graphs [5]cu.Graph
	var execs [5]cu.GraphExec
	for i, dst := range dsts {
		cu.StreamBeginCapture(captureStream, cu.STREAM_CAPTURE_MODE_THREAD_LOCAL)
		torqueFn(dst)
		graphs[i] = cu.StreamEndCapture(captureStream)
		execs[i] = graphs[i].Instantiate()
	}
	defer func() {
		for i := range execs {
			execs[i].Destroy()
			graphs[i].Destroy()
		}
	}()
	execK2, execK3, execK4, execK5, execK6 := execs[0], execs[1], execs[2], execs[3], execs[4]

	fellBack := false
	for condition() && !pause {
		if checkInject() {
			if !pause {
				fellBack = true
			}
			break
		}

		if FixDt != 0 {
			Dt_si = FixDt
		}

		t0 := Time
		data.Copy(m0, m)

		h := float32(Dt_si * GammaLL)

		// stage 2
		Time = t0 + (1./5.)*Dt_si
		cuda.Madd2(m, m, rk.k1, 1, (1./5.)*h)
		M.normalize()
		execK2.Launch(captureStream)
		NEvals++

		// stage 3
		Time = t0 + (3./10.)*Dt_si
		cuda.Madd3(m, m0, rk.k1, k2, 1, (3./40.)*h, (9./40.)*h)
		M.normalize()
		execK3.Launch(captureStream)
		NEvals++

		// stage 4
		Time = t0 + (4./5.)*Dt_si
		cuda.Madd4(m, m0, rk.k1, k2, k3, 1, (44./45.)*h, (-56./15.)*h, (32./9.)*h)
		M.normalize()
		execK4.Launch(captureStream)
		NEvals++

		// stage 5
		Time = t0 + (8./9.)*Dt_si
		cuda.Madd5(m, m0, rk.k1, k2, k3, k4, 1, (19372./6561.)*h, (-25360./2187.)*h, (64448./6561.)*h, (-212./729.)*h)
		M.normalize()
		execK5.Launch(captureStream)
		NEvals++

		// stage 6
		Time = t0 + (1.)*Dt_si
		cuda.Madd6(m, m0, rk.k1, k2, k3, k4, k5, 1, (9017./3168.)*h, (-355./33.)*h, (46732./5247.)*h, (49./176.)*h, (-5103./18656.)*h)
		M.normalize()
		execK6.Launch(captureStream)
		NEvals++

		// stage 7: 5th order solution
		Time = t0 + (1.)*Dt_si
		cuda.Madd6(m, m0, rk.k1, k3, k4, k5, k6, 1, (35./384.)*h, (500./1113.)*h, (125./192.)*h, (-2187./6784.)*h, (11./84.)*h)
		M.normalize()
		execK2.Launch(captureStream) // torqueFn(k7); k7 == k2, so execK2 applies
		NEvals++

		// error estimate
		cuda.Madd6(Err, rk.k1, k3, k4, k5, k6, k7, (35./384.)-(5179./57600.), (500./1113.)-(7571./16695.), (125./192.)-(393./640.), (-2187./6784.)-(-92097./339200.), (11./84.)-(187./2100.), (0.)-(1./40.))
		err := cuda.MaxVecNorm(Err) * float64(h)

		// adjust next time step
		if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
			// step OK
			setLastErr(err)
			NSteps++
			Time = t0 + Dt_si
			adaptDt(math.Pow(MaxErr/err, 1./5.))
			data.Copy(rk.k1, k7) // FSAL
		} else {
			// undo bad step
			util.Assert(FixDt == 0)
			Time = t0
			data.Copy(m, m0)
			NUndone++
			adaptDt(math.Pow(MaxErr/err, 1./6.))
		}

		for _, f := range postStep {
			f()
		}
		DoOutput()
	}
	pause = true
	return fellBack
}

// Implements StepsGraph/tryRunGraph for Bogacki-Shampine (solver 3),
// using the "split graph" approach (see Sec. 4.1 in You2026).
//
// Since the kernel sequence computing the torque is independent of dt and M
// (only the addresses of the buffers matter, which are fixed), it suffices to
// capture torqueFn(k2)..torqueFn(k4) once before the loop, as small graphs
// execK2..execK4. The loop body then mirrors RK23.Step exactly but with
// torqueFn calls replaced by GraphExec.Launch calls. Other functions that can
// not be captured by graphs (e.g., Madd2, MaxVecDiff...) remain interspersed
// between graph launches.
func runGraphRK23(rk *RK23, condition func() bool) bool {
	SanityCheck()
	pause = false

	rk.Free() // mirror RunWhile's stepper.Free(): start from a clean state

	m := M.Buffer()
	size := m.Size()

	// upon resize: remove wrongly sized k1
	if rk.k1.Size() != m.Size() {
		rk.Free()
	}
	// first step ever: one-time k1 init and eval
	if rk.k1 == nil {
		rk.k1 = cuda.NewSlice(3, size)
		torqueFn(rk.k1)
	}

	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)

	k2, k3, k4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)

	warmupTorque(m)

	captureStream := cuda.EnterCaptureMode()
	demagConv().SetStream(captureStream)
	defer func() {
		demagConv().SetStream(cu.Stream(0))
		cuda.ExitCaptureMode(captureStream)
	}()

	dsts := [3]*data.Slice{k2, k3, k4}
	var graphs [3]cu.Graph
	var execs [3]cu.GraphExec
	for i, dst := range dsts {
		cu.StreamBeginCapture(captureStream, cu.STREAM_CAPTURE_MODE_THREAD_LOCAL)
		torqueFn(dst)
		graphs[i] = cu.StreamEndCapture(captureStream)
		execs[i] = graphs[i].Instantiate()
	}
	defer func() {
		for i := range execs {
			execs[i].Destroy()
			graphs[i].Destroy()
		}
	}()
	execK2, execK3, execK4 := execs[0], execs[1], execs[2]

	fellBack := false
	for condition() && !pause {
		if checkInject() {
			if !pause {
				fellBack = true
			}
			break
		}

		if FixDt != 0 {
			Dt_si = FixDt
		}

		t0 := Time
		data.Copy(m0, m)

		h := float32(Dt_si * GammaLL)

		// there is no explicit stage 1: k1 from previous step

		// stage 2
		Time = t0 + (1./2.)*Dt_si
		cuda.Madd2(m, m, rk.k1, 1, (1./2.)*h) // m = m*1 + k1*h/2
		M.normalize()
		execK2.Launch(captureStream)
		NEvals++

		// stage 3
		Time = t0 + (3./4.)*Dt_si
		cuda.Madd2(m, m0, k2, 1, (3./4.)*h) // m = m0*1 + k2*3/4
		M.normalize()
		execK3.Launch(captureStream)
		NEvals++

		// 3rd order solution
		cuda.Madd4(m, m0, rk.k1, k2, k3, 1, (2./9.)*h, (1./3.)*h, (4./9.)*h)
		M.normalize()

		// error estimate
		Time = t0 + Dt_si
		execK4.Launch(captureStream)
		NEvals++
		Err := k2 // re-use k2 as error
		// difference of 3rd and 2nd order torque without explicitly storing them first
		cuda.Madd4(Err, rk.k1, k2, k3, k4, (7./24.)-(2./9.), (1./4.)-(1./3.), (1./3.)-(4./9.), (1. / 8.))

		// determine error
		err := cuda.MaxVecNorm(Err) * float64(h)

		// adjust next time step
		if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
			// step OK
			setLastErr(err)
			NSteps++
			Time = t0 + Dt_si
			adaptDt(math.Pow(MaxErr/err, 1./3.))
			data.Copy(rk.k1, k4) // FSAL
		} else {
			// undo bad step
			util.Assert(FixDt == 0)
			Time = t0
			data.Copy(m, m0)
			NUndone++
			adaptDt(math.Pow(MaxErr/err, 1./4.))
		}

		for _, f := range postStep {
			f()
		}
		DoOutput()
	}
	pause = true
	return fellBack
}

// Implements StepsGraph/tryRunGraph for Runge-Kutta-Fehlberg (solver 6),
// using the "split graph" approach (see Sec. 4.1 in You2026).
//
// Since the kernel sequence computing the torque is independent of dt and M
// (only the addresses of the buffers matter, which are fixed), it suffices to
// capture torqueFn(k2)..torqueFn(k8) once before the loop, as small graphs
// execK2..execK8. The loop body then mirrors RK56.Step exactly but with
// torqueFn calls replaced by GraphExec.Launch calls. Other functions that can
// not be captured by graphs (e.g., Madd2, MaxVecDiff...) remain interspersed
// between graph launches.
func runGraphRK56(rk *RK56, condition func() bool) bool {
	SanityCheck()
	pause = false

	rk.Free() // mirror RunWhile's stepper.Free(): start from a clean state

	m := M.Buffer()
	size := m.Size()

	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	Err := cuda.Buffer(3, size)
	defer cuda.Recycle(Err)

	k1, k2, k3, k4, k5, k6, k7, k8 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(k1)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)
	defer cuda.Recycle(k5)
	defer cuda.Recycle(k6)
	defer cuda.Recycle(k7)
	defer cuda.Recycle(k8)

	warmupTorque(m)

	captureStream := cuda.EnterCaptureMode()
	demagConv().SetStream(captureStream)
	defer func() {
		demagConv().SetStream(cu.Stream(0))
		cuda.ExitCaptureMode(captureStream)
	}()

	dsts := [8]*data.Slice{k1, k2, k3, k4, k5, k6, k7, k8}
	var graphs [8]cu.Graph
	var execs [8]cu.GraphExec
	for i, dst := range dsts {
		cu.StreamBeginCapture(captureStream, cu.STREAM_CAPTURE_MODE_THREAD_LOCAL)
		torqueFn(dst)
		graphs[i] = cu.StreamEndCapture(captureStream)
		execs[i] = graphs[i].Instantiate()
	}
	defer func() {
		for i := range execs {
			execs[i].Destroy()
			graphs[i].Destroy()
		}
	}()
	execK1, execK2, execK3, execK4, execK5, execK6, execK7, execK8 := execs[0], execs[1], execs[2], execs[3], execs[4], execs[5], execs[6], execs[7]

	fellBack := false
	for condition() && !pause {
		if checkInject() {
			if !pause {
				fellBack = true
			}
			break
		}

		if FixDt != 0 {
			Dt_si = FixDt
		}

		t0 := Time
		data.Copy(m0, m)

		h := float32(Dt_si * GammaLL)

		// stage 1
		execK1.Launch(captureStream)
		NEvals++

		// stage 2
		Time = t0 + (1./6.)*Dt_si
		cuda.Madd2(m, m, k1, 1, (1./6.)*h) // m = m*1 + k1*h/6
		M.normalize()
		execK2.Launch(captureStream)
		NEvals++

		// stage 3
		Time = t0 + (4./15.)*Dt_si
		cuda.Madd3(m, m0, k1, k2, 1, (4./75.)*h, (16./75.)*h)
		M.normalize()
		execK3.Launch(captureStream)
		NEvals++

		// stage 4
		Time = t0 + (2./3.)*Dt_si
		cuda.Madd4(m, m0, k1, k2, k3, 1, (5./6.)*h, (-8./3.)*h, (5./2.)*h)
		M.normalize()
		execK4.Launch(captureStream)
		NEvals++

		// stage 5
		Time = t0 + (4./5.)*Dt_si
		cuda.Madd5(m, m0, k1, k2, k3, k4, 1, (-8./5.)*h, (144./25.)*h, (-4.)*h, (16./25.)*h)
		M.normalize()
		execK5.Launch(captureStream)
		NEvals++

		// stage 6
		Time = t0 + (1.)*Dt_si
		cuda.Madd6(m, m0, k1, k2, k3, k4, k5, 1, (361./320.)*h, (-18./5.)*h, (407./128.)*h, (-11./80.)*h, (55./128.)*h)
		M.normalize()
		execK6.Launch(captureStream)
		NEvals++

		// stage 7
		Time = t0
		cuda.Madd5(m, m0, k1, k3, k4, k5, 1, (-11./640.)*h, (11./256.)*h, (-11/160.)*h, (11./256.)*h)
		M.normalize()
		execK7.Launch(captureStream)
		NEvals++

		// stage 8
		Time = t0 + (1.)*Dt_si
		cuda.Madd7(m, m0, k1, k2, k3, k4, k5, k7, 1, (93./640.)*h, (-18./5.)*h, (803./256.)*h, (-11./160.)*h, (99./256.)*h, (1.)*h)
		M.normalize()
		execK8.Launch(captureStream)
		NEvals++

		// stage 9: 6th order solution
		Time = t0 + (1.)*Dt_si
		cuda.Madd7(m, m0, k1, k3, k4, k5, k7, k8, 1, (7./1408.)*h, (1125./2816.)*h, (9./32.)*h, (125./768.)*h, (5./66.)*h, (5./66.)*h)
		M.normalize()
		// No need for torqueFn(k9) as k9 wouldn't be used (except in setMaxTorque, which is irrelevant)

		// error estimate
		cuda.Madd4(Err, k1, k6, k7, k8, (-5. / 66.), (-5. / 66.), (5. / 66.), (5. / 66.))
		err := cuda.MaxVecNorm(Err) * float64(h)

		// adjust next time step
		if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
			// step OK
			setLastErr(err)
			NSteps++
			Time = t0 + Dt_si
			adaptDt(math.Pow(MaxErr/err, 1./6.))
		} else {
			// undo bad step
			util.Assert(FixDt == 0)
			Time = t0
			data.Copy(m, m0)
			NUndone++
			adaptDt(math.Pow(MaxErr/err, 1./7.))
		}

		for _, f := range postStep {
			f()
		}
		DoOutput()
	}
	pause = true
	return fellBack
}

// Implements StepsGraph/tryRunGraph for backward Euler (solver -1),
// using the "split graph" approach (see Sec. 4.1 in You2026).
//
// Since the kernel sequence computing the torque is independent of dt and M
// (only the addresses of the buffers matter, which are fixed), it suffices to
// capture torqueFn(dy0) & torqueFn(dy1) once before the loop, as small graphs
// execDy0 and execDy1. The loop body then mirrors BackwardEuler.Step exactly
// but with torqueFn calls replaced by GraphExec.Launch calls. Other functions
// that can not be captured by graphs (e.g., Madd2, MaxVecDiff...) remain
// interspersed between graph launches.
//
// Note: BackwardEuler requires FixDt != 0, so dt remains constant throughout.
func runGraphBackwardEuler(s *BackwardEuler, condition func() bool) bool {
	util.AssertMsg(MaxErr > 0, "Backward euler solver requires MaxErr > 0")
	SanityCheck()
	pause = false

	s.Free() // mirror RunWhile's stepper.Free(): start from a clean state

	y := M.Buffer()

	y0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(y0)

	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)

	if s.dy1 == nil {
		s.dy1 = cuda.Buffer(VECTOR, y.Size())
	}
	dy1 := s.dy1

	warmupTorque(y)

	Dt_si = FixDt
	dt := float32(Dt_si * GammaLL)
	util.AssertMsg(dt > 0, "Backward Euler solver requires fixed time step > 0")

	captureStream := cuda.EnterCaptureMode()
	demagConv().SetStream(captureStream)
	defer func() {
		demagConv().SetStream(cu.Stream(0))
		cuda.ExitCaptureMode(captureStream)
	}()

	cu.StreamBeginCapture(captureStream, cu.STREAM_CAPTURE_MODE_THREAD_LOCAL)
	torqueFn(dy0)
	graphDy0 := cu.StreamEndCapture(captureStream)
	defer graphDy0.Destroy()
	execDy0 := graphDy0.Instantiate()
	defer execDy0.Destroy()

	cu.StreamBeginCapture(captureStream, cu.STREAM_CAPTURE_MODE_THREAD_LOCAL)
	torqueFn(dy1)
	graphDy1 := cu.StreamEndCapture(captureStream)
	defer graphDy1.Destroy()
	execDy1 := graphDy1.Instantiate()
	defer execDy1.Destroy()

	fellBack := false
	for condition() && !pause {
		if checkInject() {
			if !pause {
				fellBack = true
			}
			break
		}

		t0 := Time
		data.Copy(y0, y)

		// First guess
		Time = t0 + 0.5*Dt_si // 0.5 dt makes it implicit midpoint method

		// predictor Euler step with previous torque (Temp.isZero() guaranteed)
		cuda.Madd2(y, y0, dy1, 1, dt)
		M.normalize()

		execDy0.Launch(captureStream)
		NEvals++
		cuda.Madd2(y, y0, dy0, 1, dt) // y = y0 + dt * dy
		M.normalize()

		// One iteration
		execDy1.Launch(captureStream)
		NEvals++
		cuda.Madd2(y, y0, dy1, 1, dt) // y = y0 + dt * dy1
		M.normalize()

		Time = t0 + Dt_si

		err := cuda.MaxVecDiff(dy0, dy1) * float64(dt)

		NSteps++
		setLastErr(err)

		for _, f := range postStep {
			f()
		}
		DoOutput()
	}
	pause = true
	return fellBack
}

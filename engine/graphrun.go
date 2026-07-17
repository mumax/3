package engine

// CUDA Graph capture PoC (see CLAUDE.md Phase 1 item 3 / sections 2.7-2.10).
//
// nsys profiling showed that for small/medium grids, 75~79% of step time is
// spent in cuLaunchKernel driver calls (~27 per step), not in the kernels
// themselves. RunGraph captures the torque-evaluation kernel sequences as
// CUDA Graphs and replays them with a single cuGraphLaunch each, instead of
// ~13 individual cuLaunchKernel calls per evaluation.

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var EnableCUDAgraphs = true

func init() {
	DeclFunc("RunGraph", RunGraph, "Like Steps, but captures the torque-evaluation kernels as CUDA Graphs and "+
		"replays them for the remaining steps. Supports Heun, RK23, RK45DP (default), RK56 and BackwardEuler.")
	DeclVar("EnableCUDAgraphs", &EnableCUDAgraphs, "Enables CUDA Graphs, greatly improving performance of Run() and Steps() on small grids (default=true)<br>NOTE: graphs are only used if Temp=0, NoDemagSpins=0 and no custom/time-varying fields are defined.")
}

// RunGraph runs n further steps, capturing the GPU work of the torque
// evaluation(s) into one or more CUDA Graphs and replaying them for the
// remaining steps. Heun (solver(2)), RK23 (solver(3)), RK45DP (solver(5), the
// default), RK56 (solver(6)) and BackwardEuler (solver(-1)) are supported.
//
//   - Heun with FixDt != 0: every kernel's scalar arguments (dt, region LUTs,
//     ...) are constant from step to step (see CLAUDE.md section 2.8(d)), so
//     the whole step can be captured as a single graph and replayed unchanged
//     with GraphLaunch (runGraphFixedDt).
//   - Heun with FixDt == 0, and the other solvers (any FixDt): dt and/or the
//     accept/reject decision change every step and depend on a blocking host
//     readback (MaxVecDiff/MaxVecNorm) that stream capture cannot record (see
//     CLAUDE.md section 2.8(c)). In these cases only the dt-independent
//     torque evaluations are captured as separate small graphs ("split
//     graph"), and the dt-dependent Madd*/error-estimate/normalize/adaptDt/
//     accept-reject/FSAL steps run as ordinary (uncaptured) calls in the
//     replay loop, exactly as in the corresponding Step() (runGraphAdaptive,
//     runGraphRK45DP, runGraphRK23, runGraphRK56, runGraphBackwardEuler).
func RunGraph(n int) {
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
			fellBack = runGraphFixedDt(s, condition)
		} else {
			fellBack = runGraphAdaptive(s, condition)
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
		util.AssertMsg(false, "RunGraph: requires Heun (solver(2)), RK23 (solver(3)), RK45DP (solver(5), the default), RK56 (solver(6)), or BackwardEuler (solver(-1))")
	}
	// An Inject (GUI/script) call mid-replay may have changed state the
	// captured graphs depend on; see checkInject. Finish the remaining steps
	// the safe (non-graph) way.
	if fellBack {
		RunWhile(condition)
	}
}

// tryRunGraph attempts to run the simulation using the CUDA Graph
// capture/replay path instead of RunWhile, returning true if it did so. It
// returns false (without side effects) if the current configuration is not
// graphCompatible(), the mesh is too large to benefit (graphWorthwhile()), or
// the current solver has no graph runner -- in which case the caller (Run/
// Steps) should fall back to RunWhile(condition).
func tryRunGraph(condition func() bool) bool {
	if !graphCompatible() || !graphWorthwhile() || !EnableCUDAgraphs {
		return false
	}

	var fellBack bool
	switch s := stepper.(type) {
	case *Heun:
		if FixDt != 0 {
			fellBack = runGraphFixedDt(s, condition)
		} else {
			fellBack = runGraphAdaptive(s, condition)
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

// assertGraphCompatible panics (via util.AssertMsg) if the current
// configuration is outside the envelope RunGraph has been verified for:
// constant excitation, Temp == 0, no custom field terms, and no region-wise
// time-dependent material parameters (see CLAUDE.md section 2.11(f) and
// 2.14). Outside this envelope, a captured graph would replay a stale kernel
// sequence (fixed Time/RNG state/LUTs/etc.) every step, silently producing
// wrong results instead of erroring out.
func assertGraphCompatible() {
	if reason := graphIncompatibilityReason(); reason != "" {
		util.AssertMsg(false, "RunGraph: "+reason)
	}
}

// graphCompatible reports whether the current configuration is within the
// envelope assertGraphCompatible requires, without panicking. Used by the
// transparent Run()/Steps() graph dispatch (tryRunGraph in run.go) to decide,
// side-effect-free, whether the graph path may be used.
func graphCompatible() bool {
	return graphIncompatibilityReason() == ""
}

// graphIncompatibilityReason returns a human-readable reason why RunGraph (or
// the transparent Run()/Steps() graph path) cannot be used with the current
// configuration, or "" if it can.
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
		// blocking MemCpyDtoH in MaxVecDiff/MaxVecNorm (CLAUDE.md section 2.8(c)).
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
// Run()/Steps() graph path is not used, even if graphCompatible(). Based on
// CLAUDE.md section 2.15(c)/2.18(c)(i): at 512x512x1 (262,144 cells) RunGraph
// gives ~1.3-1.5x, at 700x700x1 (490,000 cells) ~1.14x, at 800x800x1
// (640,000 cells) ~1.06x, and at 1024x1024x1 (1,048,576 cells) only ~1.0-1.04x
// (no longer worthwhile). 800,000 includes the 800x800x1 case (~6% gain,
// "apply down to ~5% improvement" policy) while still excluding
// 1024x1024x1.
const graphMaxCells = 800000

// graphWorthwhile reports whether the current mesh is small enough for the
// transparent graph path to give a meaningful speedup (see graphMaxCells).
func graphWorthwhile() bool {
	size := Mesh().Size()
	return size[0]*size[1]*size[2] <= graphMaxCells
}

// graphMinSteps is the minimum number of steps a Steps(n)/Run(seconds) call
// must cover for the transparent graph path to be attempted. Capturing and
// instantiating the split-graph(s) has a fixed one-time cost (stream capture
// + cuGraphInstantiate per graph); this heuristic ensures it is amortized
// over enough replayed steps to be worthwhile. Tunable; not load-bearing for
// correctness (RunGraph itself has no such threshold).
const graphMinSteps = 20

// hasTimeDependentRegion reports whether p has been set, in any region, to a
// per-region time-dependent function via SetRegion/SetRegionFn with an
// expression containing t (see regionwise.upd_reg in engine/parameter.go).
func hasTimeDependentRegion(p *regionwise) bool {
	for r := 0; r < NREGION; r++ {
		if p.upd_reg[r] != nil {
			return true
		}
	}
	return false
}

// isTimeIndependent reports whether excitation e is constant in time: no
// extra mask*multiplier terms (added via Add/AddGo) and no per-region
// time-dependent value functions (e.g. set via e.SetRegion(r, vector(0, sin(t), 0))).
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

// warmupTorque performs one full torque evaluation on the normal
// (non-capturing) stream, before any CUDA Graph capture begins. SetTorque
// (via SetEffectiveField) lazily uploads region-wise material parameter LUTs
// (engine/parameter.go), computes/uploads the demag kernel on first use
// (cuda/conv_demag.go), and grows the cuda.Buffer pool to whatever scratch
// sizes the active field terms need -- all via cuMemAlloc. cuMemAlloc is not
// permitted during CUDA stream capture and crashes with Exception 0xc0000005
// if it first happens inside a captured torqueFn (CLAUDE.md section 2.16).
// This call forces all of that to happen here instead. NSteps/NEvals/Time are
// not advanced -- this calls SetTorque directly, not torqueFn.
//
// Must be called after all buffers that will be held (checked out from the
// cuda.Buffer pool) for the lifetime of the capture/replay loop (m0, Err,
// k2..k6, dy0/dy, etc.) have already been allocated -- not before. The
// cuda.Buffer pool (cuda/buffer.go) is a single stack per element count N,
// shared across all call sites; cuMemAlloc only grows it, on demand, to the
// high-water mark of buffers checked out so far. If warmupTorque ran before
// those buffers were checked out, it would only grow the pool to cover its
// own scratch + SetTorque's internal temporaries -- not enough headroom once
// m0/Err/k2..k6/etc. are ALSO checked out during the capture loop's
// torqueFn(dst) calls, so the internal temporaries would trigger a second,
// this time in-capture, cuMemAlloc and crash (CLAUDE.md section 2.16).
// Calling it last, with everything else already checked out, establishes a
// high-water mark that covers both at once.
func warmupTorque(m *data.Slice) {
	scratch := cuda.Buffer(VECTOR, m.Size())
	defer cuda.Recycle(scratch)
	SetTorque(scratch)
}

// checkInject non-blockingly checks Inject for a pending GUI/script function
// and, if present, runs it. It returns true if a function was received.
//
// All runGraph* replay loops call this once per iteration and, if it returns
// true, stop replaying captured graphs immediately (see each runGraph*'s use
// of fellBack below) instead of proceeding to that iteration's
// exec.Launch/captureStream as before. The injected function f may change
// global state the already-captured graphs depend on -- e.g. resize the mesh,
// switch solvers (stepper), or set Temp/B_ext/a material parameter to
// something graphIncompatibilityReason() would now reject -- in ways that
// graphIncompatibilityReason() does not even cover (mesh/solver changes).
// Continuing to replay the (possibly now-stale or dangling) captured graphs
// after such a change could silently produce wrong results or crash.
//
// If f left pause == true (e.g. it called Break()), the loop exits normally,
// matching runWhile's own pause handling -- no special action needed. If
// pause is still false, fellBack is set to true so the caller (RunGraph/
// tryRunGraph) finishes the remaining steps via RunWhile(condition), which
// re-derives everything from the (possibly changed) global state and is
// correct regardless of what f changed.
//
// Inject is only ever sent to from GUI-connected runs (engine/gui.go) or
// InjectAndWait (engine/render.go); RunInteractive's keepalive injector only
// runs after EvalFile returns (cmd/mumax3/main.go). So for headless runs --
// including all 176 regression tests -- checkInject always returns false and
// this entire mechanism is a no-op.
func checkInject() bool {
	select {
	case f := <-Inject:
		f()
		return true
	default:
		return false
	}
}

// runGraphFixedDt implements RunGraph for FixDt != 0: the entire step
// (StepCaptureBody) is captured once into a single graph and replayed
// unchanged for n steps.
func runGraphFixedDt(heun *Heun, condition func() bool) bool {
	SanityCheck()
	pause = false

	warmupTorque(M.Buffer())

	// Redirect stream0 (normally the NULL stream, which does not support
	// capture) to a dedicated capture/replay stream, and rebind the demag
	// FFT plans (bound to a stream once at creation time) to follow it.
	captureStream := cuda.EnterCaptureMode()
	demagConv().SetStream(captureStream)
	defer func() {
		demagConv().SetStream(cu.Stream(0))
		cuda.ExitCaptureMode(captureStream)
	}()

	cu.StreamBeginCapture(captureStream, cu.STREAM_CAPTURE_MODE_THREAD_LOCAL)
	heun.StepCaptureBody()
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
		NEvals += 2 // each replayed step performs the same 2 torque evaluations as Heun.Step
		for _, f := range postStep {
			f()
		}
		DoOutput()
	}
	pause = true
	return fellBack
}

// runGraphAdaptive implements RunGraph for FixDt == 0 (adaptive Heun).
//
// torqueFn(dy0) and torqueFn(dy) are each captured once, before the loop, as
// small graphs execT0/execT1 (the kernel sequence computing the torque from
// M is independent of dt and of M's data values -- only the addresses dy0,
// dy and M's buffer matter, and those stay fixed for the whole call). The
// loop body then mirrors Heun.Step exactly, replacing the two torqueFn calls
// with execT0.Launch/execT1.Launch and running Madd2/Madd3/MaxVecDiff/
// normalize/adaptDt as ordinary calls on the capture stream (capture has
// already ended by this point, so the blocking MemCpyDtoH inside MaxVecDiff
// is allowed).
func runGraphAdaptive(heun *Heun, condition func() bool) bool {
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

// runGraphRK45DP implements RunGraph for the default solver (RK45DP).
//
// RK45DP.Step() (engine/rk45dp.go) performs 6 torque evaluations per step:
// torqueFn(k2)..torqueFn(k6) (stages 2-6) and torqueFn(k7) (stage 7, the
// 5th-order solution), where k7 reuses k2's buffer (k7 := k2). Each
// torqueFn(kN) depends only on the addresses of M's buffer (read) and kN
// (written) -- both fixed for the lifetime of this call, and unaffected by
// dt or Time since assertGraphCompatible() guarantees a time-independent
// excitation. So each is captured once, before the loop, as a small graph
// (execK2..execK6, the "split graph" design of CLAUDE.md section 2.11(a)).
// Stage 7's torqueFn(k7) replays execK2 again, since k7 and k2 are the same
// buffer -- 5 graphs total for 6 torque evaluations.
//
// The dt-dependent Madd2..Madd6/normalize calls between torque evaluations,
// the error estimate (Madd6 into Err + MaxVecNorm, which performs the
// blocking MemCpyDtoH that stream capture cannot record, see CLAUDE.md
// section 2.8(c)), adaptDt, accept/reject, and the FSAL update
// (data.Copy(rk.k1, k7)) all run as ordinary (uncaptured) calls in the replay
// loop, mirroring RK45DP.Step() exactly. This is unchanged for FixDt == 0
// (adaptive) and FixDt != 0 alike, just as in RK45DP.Step()'s accept
// condition.
func runGraphRK45DP(rk *RK45DP, condition func() bool) bool {
	SanityCheck()
	pause = false

	rk.Free() // mirror RunWhile's stepper.Free(): start from a clean state

	m := M.Buffer()
	size := m.Size()

	// upon resize: remove wrongly sized k1 (mirrors RK45DP.Step())
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

// runGraphRK23 implements RunGraph for RK23 (Bogacki-Shampine, solver(3)).
//
// RK23.Step() (engine/rk23.go) performs 3 torque evaluations per step:
// torqueFn(k2) (stage 2), torqueFn(k3) (stage 3) and torqueFn(k4) (error
// estimate / next step's k1). Each depends only on the addresses of M's
// buffer (read) and kN (written) -- both fixed for the lifetime of this call,
// and unaffected by dt or Time since assertGraphCompatible() guarantees a
// time-independent excitation -- so each is captured once, before the loop,
// as a small graph (execK2, execK3, execK4, the "split graph" design of
// CLAUDE.md section 2.11(a)).
//
// The dt-dependent Madd2/Madd4/normalize calls between torque evaluations,
// the error estimate (Madd4 into Err + MaxVecNorm, which performs the
// blocking MemCpyDtoH that stream capture cannot record, see CLAUDE.md
// section 2.8(c)), adaptDt, accept/reject, and the FSAL update
// (data.Copy(rk.k1, k4)) all run as ordinary (uncaptured) calls in the replay
// loop, mirroring RK23.Step() exactly.
func runGraphRK23(rk *RK23, condition func() bool) bool {
	SanityCheck()
	pause = false

	rk.Free() // mirror RunWhile's stepper.Free(): start from a clean state

	m := M.Buffer()
	size := m.Size()

	// upon resize: remove wrongly sized k1 (mirrors RK23.Step())
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
		cuda.Madd4(Err, rk.k1, k2, k3, k4, (7./24.)-(2./9.), (1./4.)-(1./3.), (1./3.)-(4./9.), (1./8.))

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

// runGraphRK56 implements RunGraph for RK56 (Fehlberg, solver(6)).
//
// RK56.Step() (engine/rk56.go) performs 8 torque evaluations per step:
// torqueFn(k1)..torqueFn(k8) (stages 1-8; stage 9, the 6th-order solution,
// needs no torque evaluation). Each depends only on the addresses of M's
// buffer (read) and kN (written) -- both fixed for the lifetime of this call,
// and unaffected by dt or Time since assertGraphCompatible() guarantees a
// time-independent excitation -- so each is captured once, before the loop,
// as a small graph (execK1..execK8, the "split graph" design of CLAUDE.md
// section 2.11(a)).
//
// The dt-dependent Madd2..Madd7/normalize calls between torque evaluations,
// the error estimate (Madd4 into Err + MaxVecNorm, which performs the
// blocking MemCpyDtoH that stream capture cannot record, see CLAUDE.md
// section 2.8(c)), adaptDt and accept/reject all run as ordinary (uncaptured)
// calls in the replay loop, mirroring RK56.Step() exactly. Stage 7's
// torqueFn(k7)/execK7.Launch occurs before stage 8's Madd7, which reads k7,
// exactly as in Step().
func runGraphRK56(rk *RK56, condition func() bool) bool {
	SanityCheck()
	pause = false

	rk.Free() // mirror RunWhile's stepper.Free(): start from a clean state (no-op for RK56)

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
		cuda.Madd4(Err, k1, k6, k7, k8, (-5./66.), (-5./66.), (5./66.), (5./66.))
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

// runGraphBackwardEuler implements RunGraph for BackwardEuler (solver(-1)).
//
// BackwardEuler.Step() (engine/backwardeuler.go) performs 2 torque
// evaluations per step: torqueFn(dy0) and torqueFn(dy1). Each depends only on
// the addresses of M's buffer (read) and dyN (written) -- both fixed for the
// lifetime of this call -- so each is captured once, before the loop, as a
// small graph (execDy0, execDy1, the "split graph" design of CLAUDE.md
// section 2.11(a)).
//
// BackwardEuler requires FixDt != 0 (Dt_si = FixDt is unconditional in
// Step()), so dt is constant for the whole call. The dt-dependent
// Madd2/normalize calls, the error estimate (MaxVecDiff, which performs the
// blocking MemCpyDtoH that stream capture cannot record, see CLAUDE.md
// section 2.8(c)) and NSteps++/setLastErr all run as ordinary (uncaptured)
// calls in the replay loop, mirroring Step() exactly. The predictor step
// (Madd2(y, y0, dy1, 1, dt); M.normalize()) is unconditional here because
// assertGraphCompatible() guarantees Temp.isZero().
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

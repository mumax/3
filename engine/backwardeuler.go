package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Implicit Euler solver.
type BackwardEuler struct {
	D  *data.Slice // Diagonal approximation of the Jacobian for Newton-Raphson
	dy *data.Slice // Most recent accepted dy
}

// Euler method, can be used as solver.Step.
// Solution of the Newton-Raphson iterations follows the quasi-Newton treatment in:
//
//	Leong, W. J., Hassan, M. A., & Yusuf, M. W. (2011).
//	"A matrix-free quasi-Newton method for solving large-scale nonlinear systems".
//	Computers & Mathematics with Applications, 62(5), 2354-2363.
func (s *BackwardEuler) Step() {
	util.AssertMsg(MaxErr > 0, "Backward Euler solver requires MaxErr > 0")

	// Determine time step
	Dt_si = FixDt                                                                // 0 if adaptive time step. SI time otherwise
	dt := float32(Dt_si * GammaLL)                                               // Measure for fraction of precession [rad/T] to be stepped
	util.AssertMsg(dt > 0, "Backward Euler solver requires fixed time step > 0") // TODO: untrue, implement adaptive time stepping
	Time += Dt_si                                                                // All calculations in backward Euler use evaluations at t0 + dt

	// Backup original magnetization
	m := M.Buffer()
	size := m.Size()
	m0 := cuda.Buffer(VECTOR, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	// Upon resize: remove wrongly sized D and dy
	if s.D.Size() != size || s.dy.Size() != size {
		s.Free()
	}

	// Initialise struct attributes
	if s.D == nil { // Diagonal Jacobian starts as identity
		c := ConstVector(1, 1, 1)
		s.D = ValueOf(c)
	}
	if s.dy == nil { // Last torque
		s.dy = cuda.NewSlice(VECTOR, size)
		torqueFn(s.dy)
	} else if !Temp.isZero() { // Can not re-use last torque with temperature
		torqueFn(s.dy)
	}

	// Create buffers
	S, Ssq, dy_prev, g, g_prev_neg, tempBuf := cuda.Buffer(VECTOR, size), cuda.Buffer(VECTOR, size), cuda.Buffer(VECTOR, size), cuda.Buffer(VECTOR, size), cuda.Buffer(VECTOR, size), cuda.Buffer(VECTOR, size)
	defer cuda.Recycle(S)          // Change of m in each Newton-Raphson (NR) iteration
	defer cuda.Recycle(Ssq)        // Temporary buffer to avoid re-calculating S²
	defer cuda.Recycle(dy_prev)    // Torque of previous magnetisation state (either previous Solver or NR step)
	defer cuda.Recycle(g)          // Function evaluated by NR
	defer cuda.Recycle(g_prev_neg) // Value of NR function in previous NR iteration
	defer cuda.Recycle(tempBuf)    // CUDA array used in intermediate calculation steps

	// Iterate (quasi-Newton)
	err := MaxErr * 2 // Make sure that at least one iteration occurs
	N := 0
	for err > MaxErr {
		N += 1
		data.Copy(dy_prev, s.dy)

		// Determine g_prev_neg = -g_k(y_i)
		cuda.Madd3(g_prev_neg, m, m0, dy_prev, -1, 1, dt) // BW Euler: g_k(m_{k+1}) = m_{k+1} - m_k - h*torqueFn(t_{k+1}, m_{k+1})

		// Determine S = s_i = -(D_i)^{-1}*g_k(y_i)
		cuda.Div(S, g_prev_neg, s.D)

		// Update magnetization estimate m_{i+1} = m_{i} + S = m_{i} - (D_i)^{-1}*g_k(y_i)
		cuda.Add(m, m, S)
		M.normalize()

		// Determine g = g_k(y_{i+1})
		torqueFn(s.dy)

		// Error estimate: difference between torques in this and previous Newton iteration
		err = cuda.MaxVecDiff(s.dy, dy_prev) * float64(dt) // TODO: use err to avoid infinite loops
		if err <= MaxErr {
			break
		}

		// Update diagonal Jacobian approximation:
		//  D_{i+1} = D_i + prefactor*diag(S[i]*S[i])
		//  with prefactor = (S^T*(g_now - g_prev) - s^T*D*s) / sum(S[i]^4)
		cuda.Madd3(g, m, m0, s.dy, 1, -1, -dt) // BW Euler: g_k(m_{k+1}) = m_{k+1} - m_k - h*torqueFn(t_{k+1}, m_{k+1})
		cuda.Add(tempBuf, g, g_prev_neg)
		prefactor := cuda.Dot(S, tempBuf)
		cuda.Mul(tempBuf, s.D, S)
		prefactor -= cuda.Dot(S, tempBuf)
		cuda.Mul(Ssq, S, S)
		cuda.Mul(tempBuf, Ssq, Ssq)
		prefactor /= cuda.Sum(tempBuf.Comp(0)) + cuda.Sum(tempBuf.Comp(1)) + cuda.Sum(tempBuf.Comp(2))
		cuda.Madd2(s.D, s.D, Ssq, 1, prefactor)
	}

	NSteps++
	setLastErr(err)
	setMaxTorque(s.dy)
}

func (s *BackwardEuler) Free() {
	s.D.Free()
	s.D = nil
	s.dy.Free()
	s.dy = nil
}

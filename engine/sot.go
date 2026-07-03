package engine

// Spin-orbit (spin-Hall) torque.
//
// The spin-Hall effect in an adjacent heavy-metal layer drives damping-like and
// field-like torques on the ferromagnet. This adds them as a compensated
// effective field (see cuda/sot.cu for the derivation):
//
//	H_DL = SOT_thetaSH * (hbar * SOT_Jc) / (2 e Msat t)
//	H_FL = SOT_thetaFL * (hbar * SOT_Jc) / (2 e Msat t)
//
// Charge current SOT_Jc flows along +x with spin polarization sigma = +y (the
// standard spin-Hall geometry). SOT_thickness is the ferromagnet thickness (0
// uses the cell size along z). Use region-wise SOT_Jc to localize the current.

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	SOTthetaSH = NewScalarParam("SOT_thetaSH", "", "SOT damping-like spin-Hall angle")
	SOTthetaFL = NewScalarParam("SOT_thetaFL", "", "SOT field-like spin-Hall angle")
	SOTJc      = NewScalarParam("SOT_Jc", "A/m2", "SOT charge current density (along +x)")
	SOTthick   = NewScalarParam("SOT_thickness", "m", "Ferromagnet thickness for SOT (0 = cell size dz)")

	sotEnabled bool
	B_sot      = NewVectorField("B_sot", "T", "Spin-orbit torque effective field", AddSOTField)
)

func init() {
	DeclFunc("EnableSOT", EnableSOT, "Enable spin-orbit (spin-Hall) torque: current along +x, spin polarization +y.")
}

// EnableSOT registers the spin-orbit torque field term.
func EnableSOT() {
	if !sotEnabled {
		AddFieldTerm(B_sot)
		sotEnabled = true
	}
}

// AddSOTField adds the spin-orbit torque effective field to dst.
func AddSOTField(dst *data.Slice) {
	if !sotEnabled {
		return
	}
	thSH := float32(SOTthetaSH.GetRegion(0))
	thFL := float32(SOTthetaFL.GetRegion(0))
	if thSH == 0 && thFL == 0 {
		return
	}
	thick := float32(SOTthick.GetRegion(0))
	if thick == 0 {
		thick = float32(Mesh().CellSize()[2]) // z cell size
	}
	ms := Msat.MSlice()
	defer ms.Recycle()
	jc := SOTJc.MSlice()
	defer jc.Recycle()
	al := Alpha.MSlice()
	defer al.Recycle()
	cuda.AddSOT(dst, M.Buffer(), ms, jc, al, thSH, thFL, thick, Mesh())
}

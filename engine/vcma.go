package engine

// Voltage-controlled magnetic anisotropy (VCMA).
//
// An electric field VCMA_E across the oxide modulates the interfacial
// perpendicular anisotropy through the areal VCMA coefficient VCMA_xi:
//
//	B_z = 2 * (VCMA_xi * VCMA_E) * m_z / (Msat * VCMA_thickness)
//
// VCMA_xi is in J/(V m) (typically ~50-100 fJ/(V m)), VCMA_E in V/m, and
// VCMA_thickness is the ferromagnet/interface thickness. Perpendicular (z) easy
// axis. All parameters may be set per region.

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	VCMAxi    = NewScalarParam("VCMA_xi", "J/(V*m)", "VCMA areal coefficient")
	VCMAe     = NewScalarParam("VCMA_E", "V/m", "Perpendicular electric field for VCMA")
	VCMAthick = NewScalarParam("VCMA_thickness", "m", "Ferromagnet/interface thickness for VCMA")

	vcmaEnabled bool
	B_vcma      = NewVectorField("B_vcma", "T", "VCMA effective field", AddVCMAField)
)

func init() {
	DeclFunc("EnableVCMA", EnableVCMA, "Enable voltage-controlled magnetic anisotropy (perpendicular easy axis).")
}

// EnableVCMA enables the VCMA effective-field term.
func EnableVCMA() {
	vcmaEnabled = true
}

// AddVCMAField adds the VCMA effective field to dst.
func AddVCMAField(dst *data.Slice) {
	if !vcmaEnabled {
		return
	}
	if VCMAxi.isZero() || VCMAe.isZero() {
		return
	}
	ms := Msat.MSlice()
	defer ms.Recycle()
	e := VCMAe.MSlice()
	defer e.Recycle()
	xi := VCMAxi.MSlice()
	defer xi.Recycle()
	th := VCMAthick.MSlice()
	defer th.Recycle()
	cuda.AddVCMA(dst, M.Buffer(), ms, e, xi, th, Mesh())
}

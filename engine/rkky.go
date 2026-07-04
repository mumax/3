package engine

// Native RKKY interlayer exchange coupling.
//
// Bilinear RKKY coupling  E = -J (m1 . m2) A  between two magnetic regions,
// added as an effective-field term:
//
//	B = J / (Msat * dz) * m_partner
//
// where J is the areal coupling strength (J/m^2) and dz the cell size along z.
// J < 0 is antiferromagnetic (synthetic-antiferromagnet, SAF). The partner
// cell is located by scanning the z-column, so - unlike ext_InterExchange,
// which only couples immediately-adjacent cells - the coupling spans a
// nonmagnetic spacer gap between the layers.

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

type rkkyPair struct {
	region1, region2 int
	J                float64 // areal coupling, J/m^2
}

var (
	rkkyPairs  []rkkyPair
	B_rkky     = NewVectorField("B_rkky", "T", "RKKY interlayer coupling field", AddRKKYField)
	E_rkky     = NewScalarValue("E_rkky", "J", "RKKY interlayer coupling energy", GetRKKYEnergy)
	Edens_rkky = NewScalarField("Edens_rkky", "J/m3", "RKKY interlayer coupling energy density", AddEdens_rkky)
)

// The RKKY field is linear in m, so the standard -1/2 M.B self-energy density
// is exact (the two coupled interface cells each carry half of E = -J A m1.m2).
var AddEdens_rkky = makeEdensAdder(&B_rkky, -0.5)

func init() {
	DeclFunc("ext_RKKY", RKKY, "Adds native RKKY interlayer coupling J (J/m2) between region1 and region2 (J<0: antiferromagnetic; spans a spacer gap).")
	registerEnergy(GetRKKYEnergy, AddEdens_rkky)
}

// RKKY adds a bilinear interlayer RKKY coupling of areal strength J (J/m^2)
// between region1 and region2. J < 0 is antiferromagnetic (SAF). It may be
// called multiple times; calling it again for the same region pair overwrites
// the coupling instead of adding a duplicate.
func RKKY(region1, region2 int, J float64) {
	defRegionId(region1)
	defRegionId(region2)
	for i := range rkkyPairs {
		if (rkkyPairs[i].region1 == region1 && rkkyPairs[i].region2 == region2) ||
			(rkkyPairs[i].region1 == region2 && rkkyPairs[i].region2 == region1) {
			rkkyPairs[i].J = J
			return
		}
	}
	rkkyPairs = append(rkkyPairs, rkkyPair{region1, region2, J})
}

// GetRKKYEnergy returns the total RKKY interlayer coupling energy, in J.
func GetRKKYEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_rkky)
}

// AddRKKYField adds the RKKY field of every defined region pair to dst.
func AddRKKYField(dst *data.Slice) {
	if len(rkkyPairs) == 0 {
		return
	}
	ms := Msat.MSlice()
	defer ms.Recycle()
	m := M.Buffer()
	mesh := M.Mesh()
	reg := regions.Gpu()
	for _, p := range rkkyPairs {
		cuda.AddRKKY(dst, m, ms, reg, float32(p.J), p.region1, p.region2, mesh)
	}
}

package engine

// Native interlayer Dzyaloshinskii-Moriya interaction (chiral interlayer coupling).
//
// Interlayer-DMI coupling  E = D * zhat . (m1 x m2) * A  between two magnetic
// regions, added as an effective-field term:
//
//	B1 = +(D / (Msat*dz)) * (zhat x m2)
//	B2 = -(D / (Msat*dz)) * (zhat x m1)
//
// where D is the areal interlayer-DMI strength (J/m^2) and dz the cell size
// along z. The sign is antisymmetric in the stacking order so that the pair of
// fields is the exact variational derivative of the energy above. The partner
// cell is located by scanning the z-column, so - unlike ext_InterExchange,
// which only couples immediately-adjacent cells - the coupling spans a
// nonmagnetic spacer gap between the layers. This is the interlayer analogue of
// the intralayer interfacial DMI already in mumax3.

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

type idmiPair struct {
	region1, region2 int
	D                float64 // areal interlayer-DMI, J/m^2
}

var (
	idmiPairs  []idmiPair
	B_idmi     = NewVectorField("B_idmi", "T", "Interlayer-DMI field", AddInterlayerDMIField)
	E_idmi     = NewScalarValue("E_idmi", "J", "Interlayer-DMI energy", GetInterlayerDMIEnergy)
	Edens_idmi = NewScalarField("Edens_idmi", "J/m3", "Interlayer-DMI energy density", AddEdens_idmi)
)

// The interlayer-DMI field is the exact variational derivative of the bilinear
// energy E = D A zhat.(m1 x m2), so the standard -1/2 M.B self-energy density
// is exact.
var AddEdens_idmi = makeEdensAdder(&B_idmi, -0.5)

func init() {
	DeclFunc("ext_InterlayerDMI", InterlayerDMI, "Adds native interlayer DMI D (J/m2) between region1 and region2 (chiral interlayer coupling; spans a spacer gap).")
	registerEnergy(GetInterlayerDMIEnergy, AddEdens_idmi)
}

// InterlayerDMI adds a chiral interlayer-DMI coupling of areal strength D
// (J/m^2) between region1 and region2. It may be called multiple times; calling
// it again for the same region pair overwrites the coupling instead of adding a
// duplicate.
func InterlayerDMI(region1, region2 int, D float64) {
	defRegionId(region1)
	defRegionId(region2)
	for i := range idmiPairs {
		if (idmiPairs[i].region1 == region1 && idmiPairs[i].region2 == region2) ||
			(idmiPairs[i].region1 == region2 && idmiPairs[i].region2 == region1) {
			idmiPairs[i].D = D
			return
		}
	}
	idmiPairs = append(idmiPairs, idmiPair{region1, region2, D})
}

// GetInterlayerDMIEnergy returns the total interlayer-DMI energy, in J.
func GetInterlayerDMIEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_idmi)
}

// AddInterlayerDMIField adds the interlayer-DMI field of every defined region
// pair to dst.
func AddInterlayerDMIField(dst *data.Slice) {
	if len(idmiPairs) == 0 {
		return
	}
	ms := Msat.MSlice()
	defer ms.Recycle()
	m := M.Buffer()
	mesh := M.Mesh()
	reg := regions.Gpu()
	for _, p := range idmiPairs {
		cuda.AddInterlayerDMI(dst, m, ms, reg, float32(p.D), p.region1, p.region2, mesh)
	}
}

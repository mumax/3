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
	idmiPairs []idmiPair
	B_idmi    = NewVectorField("B_idmi", "T", "Interlayer-DMI field", AddInterlayerDMIField)
)

func init() {
	DeclFunc("ext_InterlayerDMI", InterlayerDMI, "Adds native interlayer DMI D (J/m2) between region1 and region2 (chiral interlayer coupling; spans a spacer gap).")
}

// InterlayerDMI adds a chiral interlayer-DMI coupling of areal strength D
// (J/m^2) between region1 and region2. It may be called multiple times to
// couple several region pairs.
func InterlayerDMI(region1, region2 int, D float64) {
	idmiPairs = append(idmiPairs, idmiPair{region1, region2, D})
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

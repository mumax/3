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
	rkkyPairs      []rkkyPair
	rkkyRegistered bool
	B_rkky         = NewVectorField("B_rkky", "T", "RKKY interlayer coupling field", AddRKKYField)
)

func init() {
	DeclFunc("ext_RKKY", RKKY, "Adds native RKKY interlayer coupling J (J/m2) between region1 and region2 (J<0: antiferromagnetic; spans a spacer gap).")
}

// RKKY adds a bilinear interlayer RKKY coupling of areal strength J (J/m^2)
// between region1 and region2. J < 0 is antiferromagnetic (SAF). It may be
// called multiple times to couple several region pairs.
func RKKY(region1, region2 int, J float64) {
	rkkyPairs = append(rkkyPairs, rkkyPair{region1, region2, J})
	if !rkkyRegistered {
		AddFieldTerm(B_rkky)
		rkkyRegistered = true
	}
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

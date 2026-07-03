package engine

// Native biquadratic interlayer exchange (RKKY) coupling.
//
// Bilinear + biquadratic interlayer coupling
//
//	E = -J1 (m1 . m2) A - J2 (m1 . m2)^2 A
//
// between two magnetic regions, added as an effective-field term:
//
//	B = ( J1 + 2*J2*(m.m_partner) ) / (Msat * dz) * m_partner
//
// where J1 is the bilinear and J2 the biquadratic areal coupling (J/m^2) and dz
// the cell size along z. J1<0 is antiferromagnetic; J2<0 favours the
// 90-degree (perpendicular) interlayer state characteristic of synthetic
// antiferromagnets, which the purely bilinear coupling cannot represent. With
// J2=0 this reduces to the bilinear RKKY field. The partner cell is located by
// scanning the z-column, so - unlike ext_InterExchange - the coupling spans a
// nonmagnetic spacer gap between the layers.

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

type biquadPair struct {
	region1, region2 int
	J1, J2           float64 // areal couplings, J/m^2
}

var (
	biquadPairs  []biquadPair
	B_rkkybiquad = NewVectorField("B_rkkybiquad", "T", "Biquadratic RKKY interlayer coupling field", AddBiquadraticRKKYField)
)

func init() {
	DeclFunc("ext_RKKYBiquadratic", RKKYBiquadratic, "Adds native bilinear+biquadratic RKKY coupling J1,J2 (J/m2) between region1 and region2 (J2<0 favours 90 deg; spans a spacer gap).")
}

// RKKYBiquadratic adds a bilinear + biquadratic interlayer RKKY coupling of
// areal strengths J1, J2 (J/m^2) between region1 and region2. J1 < 0 is
// antiferromagnetic; J2 < 0 favours the 90-degree state. It may be called
// multiple times to couple several region pairs.
func RKKYBiquadratic(region1, region2 int, J1, J2 float64) {
	biquadPairs = append(biquadPairs, biquadPair{region1, region2, J1, J2})
}

// AddBiquadraticRKKYField adds the biquadratic RKKY field of every defined
// region pair to dst.
func AddBiquadraticRKKYField(dst *data.Slice) {
	if len(biquadPairs) == 0 {
		return
	}
	ms := Msat.MSlice()
	defer ms.Recycle()
	m := M.Buffer()
	mesh := M.Mesh()
	reg := regions.Gpu()
	for _, p := range biquadPairs {
		cuda.AddBiquadraticRKKY(dst, m, ms, reg, float32(p.J1), float32(p.J2), p.region1, p.region2, mesh)
	}
}

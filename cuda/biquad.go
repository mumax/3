package cuda

import (
	"github.com/mumax/3/data"
)

// AddBiquadraticRKKY adds the native bilinear + biquadratic RKKY interlayer
// coupling field to B (in Tesla).
//
//	J1:               bilinear areal coupling (J/m^2); J1<0 is antiferromagnetic
//	J2:               biquadratic areal coupling (J/m^2); J2<0 favours 90 degrees
//	region1, region2: the two coupled regions
//
// The field is  B = (J1 + 2*J2*(m.mp)) / (Msat*dz) * mp, where mp is the
// partner magnetisation. With J2=0 this equals the bilinear RKKY field. The
// partner cell is located by scanning the z-column, so the coupling spans a
// nonmagnetic spacer gap between the layers. See biquad.cu.
func AddBiquadraticRKKY(B, m *data.Slice, Msat MSlice, regions *Bytes, J1, J2 float32, region1, region2 int, mesh *data.Mesh) {
	c := mesh.CellSize()
	dz := float32(c[Z])
	N := mesh.Size()
	cfg := make3DConf(N)
	k_addbiquadrkky_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		regions.Ptr, J1, J2, region1, region2,
		dz, N[X], N[Y], N[Z], cfg)
}

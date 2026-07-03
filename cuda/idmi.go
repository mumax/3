package cuda

import (
	"github.com/mumax/3/data"
)

// AddInterlayerDMI adds the native interlayer-DMI (chiral interlayer coupling)
// field to B (in Tesla).
//
//	D:                areal interlayer-DMI strength (J/m^2)
//	region1, region2: the two coupled regions
//
// The field is  B = s * (D / (Msat*dz)) * (zhat x m_partner), with the sign s
// antisymmetric in the stacking order, so that region1/region2 receive the
// exact pair of variational-derivative fields of  E = D * zhat.(m1 x m2) * A.
// The partner cell is located by scanning the z-column, so the coupling spans
// a nonmagnetic spacer gap between the layers. See idmi.cu.
func AddInterlayerDMI(B, m *data.Slice, Msat MSlice, regions *Bytes, D float32, region1, region2 int, mesh *data.Mesh) {
	c := mesh.CellSize()
	dz := float32(c[Z])
	N := mesh.Size()
	cfg := make3DConf(N)
	k_addidmi_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		regions.Ptr, D, region1, region2,
		dz, N[X], N[Y], N[Z], cfg)
}

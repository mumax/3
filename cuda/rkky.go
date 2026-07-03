package cuda

import (
	"github.com/mumax/3/data"
)

// AddRKKY adds the native RKKY interlayer coupling field to B (in Tesla).
//
//	J:                areal RKKY coupling (J/m^2); J<0 is antiferromagnetic
//	region1, region2: the two coupled regions
//
// The partner cell is located by scanning the z-column, so the coupling spans
// a nonmagnetic spacer gap between the layers. See rkky.cu.
func AddRKKY(B, m *data.Slice, Msat MSlice, regions *Bytes, J float32, region1, region2 int, mesh *data.Mesh) {
	c := mesh.CellSize()
	dz := float32(c[Z])
	N := mesh.Size()
	cfg := make3DConf(N)
	k_addrkky_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		regions.Ptr, J, region1, region2,
		dz, N[X], N[Y], N[Z], cfg)
}

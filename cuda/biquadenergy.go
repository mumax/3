package cuda

import (
	"github.com/mumax/3/data"
)

// AddBiquadraticRKKYEnergyDensity adds the biquadratic RKKY interlayer-coupling
// energy density (J/m^3) to edens, for the region1/region2 pair. See
// biquadenergy.cu. A dedicated kernel is used (rather than the generic
// -1/2 M.B density) because the biquadratic term is quartic in m.
func AddBiquadraticRKKYEnergyDensity(edens, m *data.Slice, regions *Bytes, J1, J2 float32, region1, region2 int, mesh *data.Mesh) {
	c := mesh.CellSize()
	dz := float32(c[Z])
	N := mesh.Size()
	cfg := make3DConf(N)
	k_addbiquadenergy_async(edens.DevPtr(0),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		regions.Ptr, J1, J2, region1, region2,
		dz, N[X], N[Y], N[Z], cfg)
}

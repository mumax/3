package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Add toplogogical charge density s = m · (m/∂x ❌ ∂m/∂y)
// See topologicalcharge.cu
func AddTopologicalCharge(s *data.Slice, m *data.Slice, mesh *data.Mesh) {
	cellsize := mesh.CellSize()
	N := s.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)

	k_addtopologicalcharge_async(
		s.DevPtr(X),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		float32(1.0/(cellsize[X]*cellsize[Y])), N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}

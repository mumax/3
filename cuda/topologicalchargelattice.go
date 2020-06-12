package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Set s to the toplogogical charge density for lattices
// Based on the solid angle subtended by triangle associated with three spins: a,b,c
//        s = 2 atan[(a . b x c /(1 + a.b + a.c + b.c)] / (dx dy)
// After M Boettcher et al, New J Phys 20, 103014 (2018), adapted from
// B. Berg and M. Luescher, Nucl. Phys. B 190, 412 (1981).
// A unit cell comprises two triangles, but s is a site-dependent quantity so we
// double-count and average over four triangles.
// This implementation works best for extended systems with periodic boundaries and provides a
// workable definition of the local charge density.
// See topologicalchargelattice.cu
func SetTopologicalChargeLattice(s *data.Slice, m *data.Slice, mesh *data.Mesh) {
  cellsize := mesh.CellSize()
	N := s.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)
	icxcy := float32(1.0 / (cellsize[X] * cellsize[Y]))

	k_settopologicalchargelattice_async(s.DevPtr(X),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		icxcy, N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}

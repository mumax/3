package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Add effective field due to bulk Dzyaloshinskii-Moriya interaction to Beff.
// See dmibulk.cu
func AddDMIBulk(Beff *data.Slice, m *data.Slice, Aex_red, D_red SymmLUT, Msat MSlice, regions *Bytes, mesh *data.Mesh) {
	cellsize := mesh.CellSize()
	N := Beff.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)

	k_adddmibulk_async(Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		unsafe.Pointer(Aex_red), unsafe.Pointer(D_red), regions.Ptr,
		float32(cellsize[X]), float32(cellsize[Y]), float32(cellsize[Z]), N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}

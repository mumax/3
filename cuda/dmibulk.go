package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"unsafe"
)

func AddDMIBulk(Beff *data.Slice, m *data.Slice, Aex_red, Dx_red, Dy_red, Dz_red SymmLUT, regions *Bytes, mesh *data.Mesh) {
	cellsize := mesh.CellSize()
	N := Beff.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)

	k_adddmibulk_async(Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		unsafe.Pointer(Aex_red),
		unsafe.Pointer(Dx_red), unsafe.Pointer(Dy_red), unsafe.Pointer(Dz_red),
		regions.Ptr,
		float32(cellsize[X]*1e9), float32(cellsize[Y]*1e9), float32(cellsize[Z]*1e9), N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}

package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"unsafe"
)

// Add interlayer exchange field to Beff.
// see interlayer.cu
func AddInterlayerExchange(Beff, m *data.Slice, J1_red , J2_red , toplayer, bottomlayer LUTPtr, direc LUTPtrs, regions *Bytes, mesh *data.Mesh) {
	cellsize := mesh.CellSize()
	N := Beff.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)

	k_addinterlayerexchange_async(Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		unsafe.Pointer(J1_red), unsafe.Pointer(J2_red), unsafe.Pointer(toplayer), unsafe.Pointer(bottomlayer),
		direc[X], direc[Y], direc[Z], 
		float32(cellsize[X]), float32(cellsize[Y]), float32(cellsize[Z]), N[X], N[Y], N[Z],
		regions.Ptr, cfg)
}

package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Set dst[x,y,0] to the src value in the (x,y)-column at the highest z-index still in the geometry.
// See attop.cu
func AtTopVector(dst *data.Slice, src *data.Slice, offset [3]float64, Msat MSlice, mesh *data.Mesh, vol *data.Slice) {
	cs := mesh.CellSize()
	N := src.Size()
	util.Argument(dst.NComp() == 3 && dst.Size() == [3]int{N[X], N[Y], 1})
	util.Argument(src.NComp() == 3)
	util.Argument(vol.NComp() == 1 && vol.Size() == N)
	cfg := make3DConf([3]int{N[X], N[Y], 1})

	k_attop_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		src.DevPtr(X), src.DevPtr(Y), src.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0), vol.DevPtr(0), float32(offset[X]), float32(offset[Y]),
		float32(cs[X]), float32(cs[Y]), N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}

// Set dst[x,y,0] to the src value in the (x,y)-column at the highest z-index still in the geometry.
// See attop.cu
func AtTopScalar(dst *data.Slice, src *data.Slice, offset [3]float64, Msat MSlice, mesh *data.Mesh, vol *data.Slice) {
	cs := mesh.CellSize()
	N := src.Size()
	util.Argument(dst.NComp() == 1 && dst.Size() == [3]int{N[X], N[Y], 1})
	util.Argument(src.NComp() == 1)
	util.Argument(vol.NComp() == 1 && vol.Size() == N)
	cfg := make3DConf([3]int{N[X], N[Y], 1})

	k_attop_async(dst.DevPtr(0), nil, nil, src.DevPtr(0), nil, nil,
		Msat.DevPtr(0), Msat.Mul(0), vol.DevPtr(0), float32(offset[X]), float32(offset[Y]),
		float32(cs[X]), float32(cs[Y]), N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}

package cuda

import (
//"unsafe"
"github.com/mumax/3/data"
)

// Add AFM exchange field to Beff.
// 	m: normalized magnetization
// 	B: effective field in Tesla
// 	Aex_red: Aex / (Msat * 1e18 m2)
// see exchange.cu
//func AddAFMExchange(B, m *data.Slice, AFMex float32,AFMR1 float32,AFMR2 float32,tsp float 32, regions *Bytes, mesh *data.Mesh) {
func AddAFMExchange(B, m *data.Slice, AFMex float32, AFMR1 ,AFMR2 int, tsp float32,Msat MSlice,regions *Bytes, mesh *data.Mesh) {
	c := mesh.CellSize()
	wx := float32(2 * 1e-18 / (c[X] * c[X]))
	wy := float32(2 * 1e-18 / (c[Y] * c[Y]))
	wz := float32(2 * 1e-18 / (c[Z] * c[Z]))
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	k_addAFMexchange_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		AFMex, AFMR1,AFMR2,tsp,
                Msat.DevPtr(0), Msat.Mul(0),
                regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)
//	k_addafmexchange_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
//		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
//		AFMex, AFMR1, AFMR2, regions.Ptr,
//		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)

}



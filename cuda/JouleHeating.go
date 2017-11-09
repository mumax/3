package cuda

import (
	"github.com/mumax/3/data"
)

// Thermal equation

func Evaldt0(temp0, dt0, m *data.Slice,Kth,Cth,Dth,Tsubsth,Tausubsth,res,Qext ,J MSlice,mesh *data.Mesh){
//	N := temp0.Len()
//	cfg := make1DConf(N)
//	c := mesh.CellSize()

	c := mesh.CellSize()
	N := mesh.Size()
//	pbc := mesh.PBC_code()
	cfg := make3DConf(N)

	NN := mesh.Size()

	k_evaldt0_async(
		temp0.DevPtr(0),dt0.DevPtr(0),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Kth.DevPtr(0), Kth.Mul(0),
		Cth.DevPtr(0), Cth.Mul(0),
		Dth.DevPtr(0), Dth.Mul(0),
		Tsubsth.DevPtr(0), Tsubsth.Mul(0),
		Tausubsth.DevPtr(0), Tausubsth.Mul(0),
		res.DevPtr(0), res.Mul(0),
		Qext.DevPtr(0), Qext.Mul(0),
		J.DevPtr(X), J.Mul(X),
		J.DevPtr(Y), J.Mul(Y),
		J.DevPtr(Z), J.Mul(Z),
		float32(c[X]), float32(c[Y]), float32(c[Z]), NN[X], NN[Y], NN[Z],
        	cfg)
//        	N, cfg)
}


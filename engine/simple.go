package engine

import(
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/cuda"
	"log"
)

var (
	mesh     *data.Mesh
	solver   *cuda.Heun
	Time float64
)

var (
	M Handle
)

func initialize() {
	M = NewBuffered(3, "m")

	//buffer = cuda.NewSlice(3, mesh)
	//vol = data.NilSlice(1, mesh)
	//Solver = cuda.NewHeun(m, Eval, 1e-15, Gamma0, &Time)

	//demag_ := cuda.NewDemag(mesh)
	//demag = func(dst *data.Slice) {
	//	demag_.Exec(dst, m, vol, Mu0*Msat())
	//}
	//B_demag = newHandle("B_demag")

	//exch = func(dst *data.Slice) {
	//	cuda.AddExchange(dst, m, Aex(), Mu0*Msat())
	//}
	//B_exch = newHandle("B_exch")

	//B_eff = newHandle("B_eff")

	//Torque = newHandle("torque")
}

func SetMesh(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64) {
	if mesh != nil {
		log.Fatal("mesh already set")
	}
	mesh = data.NewMesh(Nz, Ny, Nx, cellSizeZ, cellSizeY, cellSizeX)
	log.Println("set mesh:", mesh)
	initialize()
}

package gpumag

import(
	"nimble-cube/core"
	"nimble-cube/mag"
	"nimble-cube/gpu"
	"nimble-cube/gpu/conv"
)

type Demag struct{
	convolution conv.Symm2D
	b *gpu.Quant
}

//b := gpu.NewDemag(m).Output()
func NewDemag(tag string, m *gpu.Quant, accuracy... int)*Demag{
	core.Assert(m.NComp() == 3) // TODO: *Quant3
	d := new(Demag)
	acc := 9
	kernel := mag.BruteKernel(core.ZeroPad(m.Mesh), acc)
	b := gpu.NewQuant(tag, 3, m.Mesh, m.Unit()) // TODO: choose blocks
	d.convolution = *conv.NewSymm2D(m.Size(), kernel, m.MakeRChan3(), b.MakeWChan3())
}

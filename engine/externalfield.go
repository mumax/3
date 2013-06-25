package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

func init() {
	world.Var("b_ext", &B_ext)
}

var (
	B_ext     func() [3]float64 = ConstVector(0, 0, 0) // Externally applied field in T, homogeneous.
	b_ext     adderQuant
	extFields []extField
)

func initBExt() {
	b_ext = adder(3, Mesh(), "B_ext", "T", func(dst *data.Slice) {
		bext := B_ext()
		cuda.AddConst(dst, float32(bext[2]), float32(bext[1]), float32(bext[0]))
		for _, f := range extFields {
			cuda.Madd2(dst, dst, f.mask, 1, float32(f.mul()))
		}
	})
	registerEnergy(ZeemanEnergy)
}

func ZeemanEnergy() float64 {
	return -1 * Volume() * dot(&M_full, &b_ext) / Mu0
}

// Add an additional space-dependent field to B_ext.
// The field is mask * multiplier, where mask typically contains space-dependent scaling values of the order of 1.
// multiplier can be time dependent.
// TODO: extend API (set one component, construct masks or read from file). Also for current.
func AddExtField(mask *data.Slice, multiplier func() float64) {
	m := cuda.GPUCopy(mask)
	extFields = append(extFields, extField{m, multiplier})
}

type extField struct {
	mask *data.Slice
	mul  func() float64
}

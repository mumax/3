package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

type excitation struct {
	VectorParam
	extraTerms []mulmask
	adderQuant
}

type mulmask struct {
	mul  func() [3]float64
	mask *data.Slice
}

var (
	B_ext excitation
)

func init() {
	world.Var("B_ext", &B_ext)
}

func initBExt() {
	B_ext.adderQuant = adder(3, Mesh(), "B_ext", "T", func(dst *data.Slice) {

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

//// Add an additional space-dependent field to B_ext.
//// The field is mask * multiplier, where mask typically contains space-dependent scaling values of the order of 1.
//// multiplier can be time dependent.
//// TODO: extend API (set one component, construct masks or read from file). Also for current.
//func AddExtField(mask *data.Slice, multiplier func() float64) {
//	m := cuda.GPUCopy(mask)
//	extFields = append(extFields, extField{m, multiplier})
//}
//
//type extField struct {
//	mask *data.Slice
//	mul  func() float64
//}

package mx

import ()

type Scalar interface {
	Quant
	IGet1(index int) float32
}

type UniformScalar interface {
	Quant
	Get1() float32
}

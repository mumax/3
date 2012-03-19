package mx

import ()

type Quant interface {
	Name() string
	Unit() string
	NComp() int
	IGet(comp, index int) float32
}

// Uniform quantity is uniform over space.
type Uniform interface {
	Quant
	Get(comp int) float32
}

type Scalar interface {
	Quant
	IGet1(index int) float32
}

type UniformScalar interface {
	Quant
	Get1() float32
}

type Vector interface {
	Quant
	IGet3(index int) [3]float32
}

type UniformVector interface {
	Quant
	IGet3(index int) [3]float32
}

package mx

import ()

// Uniform quantity is uniform over space.
type Uniform interface {
	Quant
	Get(comp int) float32
}

// UniformN is an in-memory N-component uniform quantity.
type UniformN struct {
	AQuant
	value []float32
}

func (this *UniformN) Init(value []float32) {
	this.value = value
}

func NewUniform(value []float32) *UniformN {
	this := new(UniformN)
	this.Init(value)
	return this
}

func NewUniformScalar(value float32) *UniformN {
	return NewUniform([]float32{value})
}

func (this *UniformN) IGet(comp, index int) float32 {
	return this.value[comp]
}

func (this *UniformN) NComp() int {
	return len(this.value)
}

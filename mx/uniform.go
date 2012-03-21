package mx

import ()

// UniformN is an in-memory N-component uniform quantity.
type UniformN struct {
	AQuant
	value []float32
}

func (this *UniformN) Init(value []float32) {
	this.value = make([]float32, len(value))
	copy(this.value, value)
}

func NewUniform(value []float32) *UniformN {
	this := new(UniformN)
	this.Init(value)
	return this
}

// Implements Uniform.
func (this *UniformN) Get(comp int) float32 {
	return this.value[comp]
}

// Implements Quant.
func (this *UniformN) IGet(comp, index int) float32 {
	return this.value[comp]
}

// Implements Quant.
func (this *UniformN) NComp() int {
	return len(this.value)
}

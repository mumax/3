package main

import ()

// Uniform quantity is uniform over space.
type Uniform interface {
	Quant
	GetUniform(comp int) float32
}

type UniformScalar interface{
	Uniform
	Scalar
}

// UniformN is an in-memory N-component uniform quantity.
type UniformN struct {
	AQuant
	value []float32
}

func (this *UniformN) Init(value []float32) {
	this.AQuant.Init()
	this.value = value
}

func NewUniform(value []float32) *UniformN {
	this := new(UniformN)
	this.Init(value)
	return this
}

func NewUniformScalar(value float32) *UniformN{
	return NewUniform([]float32{value})
}

func (this *UniformN) Get(comp, index int) float32 {
	return this.value[comp]
}

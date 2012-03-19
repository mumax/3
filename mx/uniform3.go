package mx

import ()

// Uniform3 is an in-memory uniform vector quantity.
type Uniform3 struct {
	AQuant
	value [3]float32
}

func (this *Uniform3) Init(value [3]float32) {
	this.value = value
}

func NewUniformVector(value [3]float32) *Uniform3 {
	this := new(Uniform3)
	this.Init(value)
	return this
}

// Implements UniformVector.
func (this *Uniform3) Get3() [3]float32 {
	return this.value
}

// Implements Uniform
func (this *Uniform3) Get(comp int) float32 {
	return this.value[comp]
}

// Implements Vector
func (this *Uniform3) IGet3(index int) [3]float32 {
	return this.value
}

// Implements Quant
func (this *Uniform3) IGet(comp, index int) float32 {
	return this.value[comp]
}

// Implements Quant
func (this *Uniform3) NComp() int {
	return 3
}

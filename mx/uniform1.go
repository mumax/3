package mx

import ()

// Uniform1 is an in-memory uniform scalar quantity.
type Uniform1 struct {
	AQuant
	value [1]float32 // use array instead of number for in-lined comp bound check
}

func (this *Uniform1) Init(value float32) {
	this.value[0] = value
}

func NewUniformScalar(value float32) *Uniform1 {
	this := new(Uniform1)
	this.Init(value)
	return this
}

// Implements UniformScalar
func (this *Uniform1) Get1() float32 {
	return this.value[0]
}

// Implements Uniform
func (this *Uniform1) Get(comp int) float32 {
	if comp != 0 {
		panic("comp out of range")
	}
	return this.value[0]
}

// Implements Scalar
func (this *Uniform1) IGet1(index int) float32 {
	return this.value[0]
}

// Implements Quant
func (this *Uniform1) IGet(comp, index int) float32 {
	return this.value[comp] // this checks comp's bounds while still inlining the possible panic.
}

// Implements Quant
func (this *Uniform1) NComp() int {
	return 1
}

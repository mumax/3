package mx

import ()

// Uniform1 is an in-memory uniform scalar quantity.
type Uniform1 struct {
	AQuant
	value [1]float32 // use array instead of number for in-lined comp bound check
	Buffer
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
	return this.value[comp]
}

// Implements Scalar
func (u *Uniform1) IGet1(i1, i2 int) []float32 {
	return u.IGet(0, i1, i2)
}

// Implements Quant
func (u *Uniform1) IGet(comp, i1, i2 int) []float32 {
	// Buffer has already correct length: it is initialized
	if len(u.Buffer) == i2-i1 {
		return u.Buffer
	}
	u.MakeBuffer(i2 - i1)
	for i := range u.Buffer {
		u.Buffer[i] = u.value[0]
	}
	return u.Buffer
}

// Implements Quant
func (this *Uniform1) NComp() int {
	return 1
}

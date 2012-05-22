package mx

import ()

// UniformN is an in-memory N-component uniform quantity.
type UniformN struct {
	AQuant
	value []float32
	buf   []Buffer
}

func (u *UniformN) Init(value []float32) {
	nComp := len(value)
	u.value = make([]float32, nComp)
	copy(u.value, value)
	u.buf = make([]Buffer, nComp)
}

func NewUniform(value []float32) *UniformN {
	u := new(UniformN)
	u.Init(value)
	return u
}

// Implements Uniform.
func (u *UniformN) Get(comp int) float32 {
	return u.value[comp]
}

// Implements Quant.
func (u *UniformN) IGet(comp, i1, i2 int) []float32 {
	// Buffer has already correct length: it is initialized
	if len(u.buf[comp]) == i2-i1 {
		return u.buf[comp]
	}
	u.buf[comp].MakeBuffer(i2 - i1)
	for i := range u.buf[comp] {
		u.buf[comp][i] = u.value[comp]
	}
	return u.buf[comp]
}

// Implements Quant.
func (u *UniformN) NComp() int {
	return len(u.value)
}

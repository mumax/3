package mx

import ()

// UniformN is an in-memory N-component uniform quantity.
type UniformN struct {
	AQuant
	value []float32
}

func (this *UniformN) Init(value []float32) { //←[ can inline (*UniformN).Init  (*UniformN).Init this does not escape  (*UniformN).Init value does not escape]
	this.value = make([]float32, len(value)) //←[ make([]float32, len(value)) escapes to heap]
	copy(this.value, value)
}

func NewUniform(value []float32) *UniformN { //←[ NewUniform value does not escape]
	this := new(UniformN) //←[ new(UniformN) escapes to heap]
	this.Init(value)      //←[ inlining call to (*UniformN).Init  make([]float32, len(value)) escapes to heap]
	return this
}

// Implements Uniform.
func (this *UniformN) Get(comp int) float32 { //←[ can inline (*UniformN).Get  (*UniformN).Get this does not escape]
	return this.value[comp]
}

// Implements Quant.
func (this *UniformN) IGet(comp, index int) float32 { //←[ can inline (*UniformN).IGet  (*UniformN).IGet this does not escape]
	return this.value[comp]
}

// Implements Quant.
func (this *UniformN) NComp() int { //←[ can inline (*UniformN).NComp  (*UniformN).NComp this does not escape]
	return len(this.value)
}

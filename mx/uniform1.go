package mx

import ()

// Uniform1 is an in-memory uniform scalar quantity.
type Uniform1 struct {
	AQuant
	value [1]float32 // use array instead of number for in-lined comp bound check
}

func (this *Uniform1) Init(value float32) { //←[ can inline (*Uniform1).Init  (*Uniform1).Init this does not escape]
	this.value[0] = value
}

func NewUniformScalar(value float32) *Uniform1 {
	this := new(Uniform1) //←[ new(Uniform1) escapes to heap]
	this.Init(value)      //←[ inlining call to (*Uniform1).Init]
	return this
}

// Implements UniformScalar
func (this *Uniform1) Get1() float32 { //←[ can inline (*Uniform1).Get1  (*Uniform1).Get1 this does not escape]
	return this.value[0]
}

// Implements Uniform
func (this *Uniform1) Get(comp int) float32 { //←[ (*Uniform1).Get this does not escape]
	if comp != 0 {
		panic("comp out of range")
	}
	return this.value[0]
}

// Implements Scalar
func (this *Uniform1) IGet1(index int) float32 { //←[ can inline (*Uniform1).IGet1  (*Uniform1).IGet1 this does not escape]
	return this.value[0]
}

// Implements Quant
func (this *Uniform1) IGet(comp, index int) float32 { //←[ can inline (*Uniform1).IGet  (*Uniform1).IGet this does not escape]
	return this.value[comp] // this checks comp's bounds while still inlining the possible panic.
}

// Implements Quant
func (this *Uniform1) NComp() int { //←[ can inline (*Uniform1).NComp  (*Uniform1).NComp this does not escape]
	return 1
}

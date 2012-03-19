package mx

import ()

// Uniform3 is an in-memory uniform vector quantity.
type Uniform3 struct {
	AQuant
	value [3]float32
}

func (this *Uniform3) Init(value [3]float32) { //←[ can inline (*Uniform3).Init  (*Uniform3).Init this does not escape]
	this.value = value
}

func NewUniformVector(value [3]float32) *Uniform3 {
	this := new(Uniform3) //←[ new(Uniform3) escapes to heap]
	this.Init(value)      //←[ inlining call to (*Uniform3).Init]
	return this
}

// Implements UniformVector.
func (this *Uniform3) Get3() [3]float32 { //←[ can inline (*Uniform3).Get3  (*Uniform3).Get3 this does not escape]
	return this.value
}

// Implements Uniform
func (this *Uniform3) Get(comp int) float32 { //←[ can inline (*Uniform3).Get  (*Uniform3).Get this does not escape]
	return this.value[comp]
}

// Implements Vector
func (this *Uniform3) IGet3(index int) [3]float32 { //←[ can inline (*Uniform3).IGet3  (*Uniform3).IGet3 this does not escape]
	return this.value
}

// Implements Quant
func (this *Uniform3) IGet(comp, index int) float32 { //←[ can inline (*Uniform3).IGet  (*Uniform3).IGet this does not escape]
	return this.value[comp]
}

// Implements Quant
func (this *Uniform3) NComp() int { //←[ can inline (*Uniform3).NComp  (*Uniform3).NComp this does not escape]
	return 3
}

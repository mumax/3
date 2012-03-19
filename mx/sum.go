package mx

import ()

type Sum struct {
	AQuant
	a, b Quant
}

func NewSum(a, b Quant) *Sum {
	this := new(Sum)
	CheckUnits(a.Unit(), b.Unit())
	CheckNComp(a.NComp(), b.NComp())
	this.a = a
	this.b = b
	this.name = "(" + a.Name() + "+" + b.Name() + ")"

	return this
}

func (this *Sum) IGet(comp, index int) float32 {
	/*var A, B float32

	switch a := this.a.(type) {
	default:
		A = a.IGet(comp, index)
	case *Uniform1:
		A = a.IGet(comp, index)
	case *Uniform3:
		A = a.IGet(comp, index)
	case *UniformN:
		A = a.IGet(comp, index)
	}

	switch b := this.b.(type) {
	default:
		B = b.IGet(comp, index)
	case *Uniform1:
		B = b.IGet(comp, index)
	case *Uniform3:
		B = b.IGet(comp, index)
	case *UniformN:
		B = b.IGet(comp, index)
	}
	return A + B*/
	return this.a.IGet(comp, index) + this.b.IGet(comp, index)
}

func (this *Sum) NComp() int {
	return this.a.NComp()
}

func (this *Sum) Update() {
	println("nop")
}

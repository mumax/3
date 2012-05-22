package mx

import ()

type Sum struct {
	AQuant
	Buffer32
	a, b Quant
}

func NewSum(a, b Quant) *Sum {
	this := new(Sum)

	CheckNComp(a.NComp(), b.NComp())

	this.a = a
	this.b = b

	CheckUnits(a.Unit(), b.Unit())
	this.unit = a.Unit()
	this.name = "(" + a.Name() + "+" + b.Name() + ")"

	return this
}

func (this *Sum) IGet(comp, i1, i2 int) []float32 {
	a := this.a.IGet(comp, i1, i2)
	b := this.b.IGet(comp, i1, i2)
	c := this.Buffer(i2 - i1)
	for i := range c {
		c[i] = a[i] + b[i]
	}
	return c
}

func (this *Sum) NComp() int {
	return this.a.NComp()
}

func (this *Sum) Update() {
	println("nop")
}

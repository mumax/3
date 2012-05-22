package mx

import ()

type Sum struct {
	AQuant
	Buffer
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

func (s *Sum) IGet(comp, i1, i2 int) []float32 {
	a := s.a.IGet(comp, i1, i2)
	b := s.b.IGet(comp, i1, i2)
	s.MakeBuffer(i2 - i1)
	for i := range s.Buffer {
		s.Buffer[i] = a[i] + b[i]
	}
	return s.Buffer
}

func (this *Sum) NComp() int {
	return this.a.NComp()
}

//func (this *Sum) Update() {
//	println("nop")
//}

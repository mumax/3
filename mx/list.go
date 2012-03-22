package mx

import ()

// List is an N-component quantity backed by an in-memory array.
type List struct {
	AQuant
	Data []float32   // Underlying contiguous storage
	Comp [][]float32 // Individual components
}

func (this *List) Init(nComp, length int) {
	this.Data, this.Comp = Array2D(nComp, length)
}

func NewList(nComp, length int) *List {
	this := new(List)
	this.Init(nComp, length)
	return this
}

func (this *List) IGet(comp, index int) float32 {
	return this.Comp[comp][index]
}

func (this *List) NComp() int {
	return len(this.Comp)
}

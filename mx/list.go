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

//// Allocates an returns a new Array
//func NewArray(components int, size3D []int) *Array {
//	t := new(Array)
//	t.Init(components, size3D)
//	return t
//}
//
//func (a *Array) Rank() int {
//	return len(a.Size)
//}
//
//func (a *Array) Len() int {
//	return a.Size[0] * a.Size[1] * a.Size[2] * a.Size[3]
//}
//
//func (a *Array) NComp() int {
//	return a.Size[0]
//}
//
//// Component array, shares storage with original
//func (a *Array) Component(component int) *Array {
//	comp := new(Array)
//	copy(comp.Size[:], a.Size[:])
//	comp.Size[0] = 1 // 1 component
//	comp.Size4D = comp.Size[:]
//	comp.Size3D = comp.Size[1:]
//	comp.List = a.Comp[component]
//	comp.Array = Slice4D(comp.List, comp.Size4D)
//	comp.Comp = Slice2D(comp.List, []int{1, Prod(comp.Size3D)})
//	return comp
//}

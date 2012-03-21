package mx

import ()

//// Array is an N-component quantity backed by an in-memory array.
//type Array struct {
//	List
//	Array  [][][][]float32 // Array in the usual way
//	Size3D [3]int          // {size0, size1, size2}
//}
//
//// Initializes a pre-allocated Array struct
//func (this *Array) Init(nComp int, size3D []int) {
//	Assert(len(size3D) == 3)
//	this.List.Init(nComp, prod(size3D))
//	//, this.Array = Array4D(nComp, size3D[0], size3D[1], size3D[2])
//	copy(this.Size3D[:], size3D)
//}
//
////// Allocates an returns a new Array
////func NewArray(components int, size3D []int) *Array {
////	t := new(Array)
////	t.Init(components, size3D)
////	return t
////}
////
////func (a *Array) Rank() int {
////	return len(a.Size)
////}
////
////func (a *Array) Len() int {
////	return a.Size[0] * a.Size[1] * a.Size[2] * a.Size[3]
////}
////
////func (a *Array) NComp() int {
////	return a.Size[0]
////}
////
////// Component array, shares storage with original
////func (a *Array) Component(component int) *Array {
////	comp := new(Array)
////	copy(comp.Size[:], a.Size[:])
////	comp.Size[0] = 1 // 1 component
////	comp.Size4D = comp.Size[:]
////	comp.Size3D = comp.Size[1:]
////	comp.List = a.Comp[component]
////	comp.Array = Slice4D(comp.List, comp.Size4D)
////	comp.Comp = Slice2D(comp.List, []int{1, Prod(comp.Size3D)})
////	return comp
////}

package mx

import ()

// List is an N-component quantity backed by an in-memory array.
type VectorList struct {
	List
}

func (this *VectorList) Init(length int) {
	this.List.Init(1, length)
}

func NewVectorList(length int) *VectorList {
	this := new(VectorList)
	this.Init(length)
	return this
}

//func (this *VectorList) IGet3(index int) [3]float32 {
//	return [3]float32{this.IGet(0, index), this.IGet(1, index), this.IGet(2, index)}
//}

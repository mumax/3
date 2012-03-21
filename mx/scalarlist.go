package mx

import ()

// List is an N-component quantity backed by an in-memory array.
type ScalarList struct {
	List
}

func (this *ScalarList) Init(length int) {
	this.List.Init(1, length)
}

func NewScalarList(length int) *ScalarList {
	this := new(ScalarList)
	this.Init(length)
	return this
}

func (this *ScalarList) IGet1(index int) float32 {
	return this.IGet(0, index)
}

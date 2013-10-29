package data

import (
	"testing"
)

func TestIndex(t *testing.T) {
	mesh := NewMesh(4, 5, 6, 1e-9, 2e-9, 3e-9)
	slice := NewSlice(3, mesh)
	data := slice.Tensors()

	if len(data) != 3 {
		t.Fail()
	}
	if len(data[0]) != 6 {
		t.Fail()
	}
	if len(data[0][0]) != 5 {
		t.Fail()
	}
	if len(data[0][0][0]) != 4 {
		t.Fail()
	}
}

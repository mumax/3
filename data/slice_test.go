package data

import (
	"testing"
)

func TestIndex(t *testing.T) {
	mesh := NewMesh(6, 5, 4, 1e-9, 2e-9, 3e-9)
	slice := NewSlice(3, mesh)
	data := slice.Tensors()

	if len(data) != 3 { //c
		t.Fail()
	}
	if len(data[0]) != 4 { // z
		t.Fail()
	}
	if len(data[0][0]) != 5 { // y
		t.Fail()
	}
	if len(data[0][0][0]) != 6 { // x
		t.Fail()
	}

	slice.Set(2, 5, 4, 3, 345) // c x y z
	if data[2][3][4][5] != 345 {
		t.Fail()
	}
}

package data

import (
	"testing"
)

func TestIndex(t *testing.T) {
	mesh := [3]int{6, 5, 4}
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
